from typing import Optional, List, Any, Dict, Tuple
import sys
import time

import pandas as pd
import onnxruntime_genai as og
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoTokenizer,
)
from colorama import Fore, Style, init

sys.path.append(".")
from src.utils.threads import KillableThread
from src.utils.custom_classes import StopWordCriteria
from src.finetuning.helper_functions import clean_gpu

init(autoreset=True)


def is_flash_attention_available() -> bool:
    try:
        from flash_attn import flash_attn_fn

        return True
    except ImportError:
        return False


def calculate_confidence(logits: torch.Tensor) -> float:
    """
    Calculates confidence as the maximum probability from the softmax distribution.
    Output is a value between 0 and 1, where 1 is the most confident.
    """
    probs = torch.softmax(logits, dim=-1)
    max_prob = torch.max(probs).item()
    return max_prob


def get_color_based_on_confidence(confidence: float) -> str:
    """
    Returns a color string based on the confidence score.
    Scale from red (low confidence) to green (high confidence).
    """
    red = int((1 - confidence) * 255)
    green = int(confidence * 255)
    return f"\033[38;2;{red};{green};0m"


def format_confidence_table(df: pd.DataFrame) -> str:
    """
    Formats a DataFrame containing 'text' and 'confidence' columns into a table string.

    Parameters:
    df (pd.DataFrame): DataFrame with 'text' and 'confidence' columns.

    Returns:
    str: A formatted string representing the table.
    """
    table_str = (
        "| Row | text           | confidence |\n|-----|----------------|------------|\n"
    )
    for i, row in df.iterrows():
        text = f'"{row["text"]}"' if pd.notna(row["text"]) else "<NA>"
        confidence = (
            f'{row["confidence"]:.6f}' if pd.notna(row["confidence"]) else "<NA>"
        )
        table_str += f"| {i:<3} | {text:<14} | {confidence:<10} |\n"
    return table_str


class TextGenerationProcessor:
    def __init__(
        self,
        model_id: str,
        device: str = None,
        quantization_bits: Optional[int] = None,
        bnb_kwargs: Optional[dict] = None,
        gguf_file: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ):
        """
        Initializes the TextGenerationProcessor, which is a wrapper around language models.

        Parameters
        ----------
        model_id : str
            The model ID or path to use.
        device : str, optional
            The device to run the model on. If None, it will default to "cuda" if a GPU is available, otherwise "cpu".
        quantization_bits : int, optional
            Only applies to GPUs. Used to quantize the model weights to reduce memory usage and increase inference speed.
            4-bit or 8-bit quantization can be applied, by setting this parameter to 4 or 8, respectively.
        bnb_kwargs : dict, optional
            Additional kwargs for BitsAndBytesConfig.
        gguf_file : str, optional
            Name of the GGUF file to load. This file should be located in the same directory as the model_id.
        tokenizer_kwargs : dict, optional
            Additional kwargs for AutoTokenizer.from_pretrained().
        model_kwargs : dict, optional
            Additional kwargs for AutoModelForCausalLM.from_pretrained().
        """
        if quantization_bits is not None and quantization_bits not in [4, 8]:
            raise ValueError("Quantization bits must be either None, 4 or 8.")

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._attention_implementation = (
            "flash" if is_flash_attention_available() else "eager"
        )

        tokenizer_kwargs = tokenizer_kwargs or {}
        tokenizer_kwargs.setdefault("trust_remote_code", True)
        tokenizer_kwargs.setdefault("use_fast", True)

        if gguf_file:
            tokenizer_kwargs["gguf_file"] = gguf_file

        # tokenizer and model are accesible by pipe.tokenizer and pipe.model
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        model = self._load_model(
            quantization_bits,
            gguf_file=gguf_file,
            bnb_kwargs=bnb_kwargs or {},
            model_kwargs=model_kwargs or {},
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if self.device != "cpu" else {"": "cpu"},
        )
        self.streamer = TextIteratorStreamer(
            self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self._confidence_stats: List[Tuple[str, float]] = []

    @property
    def confidence_stats(self) -> pd.DataFrame:
        return pd.DataFrame(self._confidence_stats, columns=["text", "confidence"])

    def _load_model(
        self,
        quantization_bits: Optional[int],
        gguf_file: Optional[str],
        bnb_kwargs: Dict,
        model_kwargs: Dict,
    ):
        quantization_config = None
        if quantization_bits == 4 and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                **bnb_kwargs,
            )
        elif quantization_bits == 8 and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, **bnb_kwargs)

        model_kwargs.update(
            {
                "device_map": "auto" if self.device != "cpu" else {"": "cpu"},
                "trust_remote_code": True,
                "torch_dtype": (
                    "auto" if self.device != "cpu" else torch.float32
                ),  # cpu only supports float32
                "quantization_config": quantization_config,
                "_attn_implementation": self._attention_implementation,
            }
        )

        if gguf_file:
            model_kwargs["gguf_file"] = gguf_file

        return AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

    @torch.inference_mode()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        add_confidence: bool = False,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> str:
        self._confidence_stats = []
        _generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "streamer": self.streamer,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0.0 else None,
            "do_sample": temperature > 0.0,
        }

        if stopwords is not None:
            _generation_kwargs["stopping_criteria"] = (
                StopWordCriteria(
                    prompts=[input_text],
                    stop_words=stopwords,
                    tokenizer=self.pipe.tokenizer,
                ),
            )

        if generation_kwargs is not None:
            _generation_kwargs.update(generation_kwargs)

        generation_thread = KillableThread(
            target=self.pipe, args=(input_text,), kwargs=_generation_kwargs
        )
        generation_thread.start()

        generated_text = ""
        for new_text in self.streamer:
            if add_confidence:
                try:
                    # Calculate confidence for the last token
                    input_ids = self.pipe.tokenizer.encode(
                        new_text, return_tensors="pt"
                    ).to(self.device)
                    last_token_logits = self.pipe.model(input_ids=input_ids).logits[
                        :, -1, :
                    ]
                    confidence = calculate_confidence(last_token_logits)
                    self._confidence_stats.append((new_text, confidence))
                    color = get_color_based_on_confidence(confidence)
                    new_text = f"{color}{new_text}{Style.RESET_ALL}"
                except RuntimeError:
                    self._confidence_stats.append((new_text, pd.NA))
                    new_text = f"{Fore.WHITE}{new_text}{Style.RESET_ALL}"

            generated_text += new_text
            print(new_text, end="", flush=True)

        print()  # Print newline after generation is done

        generation_thread.kill()
        generation_thread.join()

        return generated_text

