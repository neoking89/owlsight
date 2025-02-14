# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models.
It offers an interactive interface that allows you to execute Python code, shell commands, and use an AI assistant in one unified environment. 
Next to that, Owlsight offers an extensive set of tools in its backend-API, which enables you to use most of the existing CLI-functionaliy in your own Python scripts.

## Why owlsight?

Picture this: you are someone who dabbles in Python occasionally. Or you frequently use generative AI to accelerate your workflow, whether for generating code or working with data.
Often, this involves a tedious process—copying and pasting code between ChatGPT and your IDE, repeatedly switching contexts.

What if you could eliminate this friction?

Owlsight brings Python and generative AI together in an intuitive Command Line Interface, streamlining your workflow by integrating them into a single, unified platform. 
No more toggling between windows, no more manual code transfers. With Owlsight, you get the full power of Python and AI, all in one place—simplifying your process and boosting productivity. 
Owlsight has been designed to be a swiss-army knife for Python and AI with a core focus on open-source models, allowing you to execute code directly from model prompts and access this code directly from the Python interpreter.

## Installation of the CLI:

You can install Owlsight using pip:

```bash
pip install owlsight
```

By default, only the transformers library is installed for working with language models.

To add GGUF functionality:
```
pip install owlsight[gguf]
```

To add ONNX functionality:
```
pip install owlsight[onnx]
```

To add multimodal functionality:
```
pip install owlsight[multimodal]
```

When working offline, you can use the offline flag. 
This will enable access to the tika-server.jar file locally, enabling you to use the `DocumentReader` class (which includes Apache Tika functionality) without an internet connection.
```
pip install owlsight[offline]
```

To install all packages:
```
pip install owlsight[all]
```

It is recommended to use the `all` option, as this will install all dependencies and allow you to use all features of Owlsight.

NOTE: some libraries like llama-cpp-python and pytorch can be highly dependant on user-specific configurations.
From Owlsight out of the box, these libraries are installed without any additional configurations.
You might need to reinstall them after installing Owlsight with settings that match your requirements.

## Usage

After installation, launch Owlsight in the terminal by running the following command:

```
owlsight
```

This will present you with some giant ASCII-art of an owl and information which tells you whether you have access to an active GPU (assuming you use CUDA).

Then, you are presented with the mainmenu:

```
Current choice:
> how can I assist you?
shell
python
config: main
save
load
clear history
quit
```

A choice can be made in the mainmenu by pressing the UP and DOWN arrow keys.
In the config menu, the LEFT and RIGHT arrow keys can be used to navigate between the different sections.
From the config sections, press "back" to go back to the mainmenu.
Press ENTER to select an option.
If you want to change an option, press ENTER to confirm the change.

### Keyboard Shortcuts

When working with the editable option, the following keyboard shortcuts are available:

- **Ctrl+A**: Select all text in the current editable field
- **Ctrl+C**: Copy selected text
- **Ctrl+Y**: Paste selected text

### Getting Started

Now, lets start out by loading a model. Go to **config** and toggle a few times to the right to reach the **huggingface** section. Choose a task like *text-generation* and press ENTER. 

Then, use the *search* option to search for a model. 
You can first type in keywords before searching, like "llama gguf". This will give you results from the Huggingface modelhub which are related to models in the llama-family in GGUf format.

Press ENTER to see the top_k results. Use the LEFT and RIGHT arrow keys in the *select_model* option to select a model and press ENTER to load it.

### Available Commands

The following available commands are available from the mainmenu:

* **How can I assist you**: Ask a question or give an instruction. By default, model responses are streamed to the console.
* **shell** : Execute shell commands. This can be useful for pip installing python libraries inside the application.
* **python** : Enter a Python interpreter. Press exit() to return to the mainmenu.
* **config: main** : Modify the *main*, *model* , *generate* or *rag* configuration settings.
* **save/load** : Save or load a configuration file.
* **clear history** : Clear the chat history and cache folder.
* **quit** : Exit the application.

### Example Workflow

You can combine Python variables with language models in Owlsight through special double curly-brackets syntax. For example:

```
python > a = 42
How can I assist you? > How much is {{a}} * 5?
```

```
answer -> 210
```

Additionally, you can also ask a model to write pythoncode and access that in the python interpreter.

From a model response, all generated python code will be extracted and can be edited or executed afterwards. This choice is always optional. After execution, the defined objects will be saved in the global namespace of the python interpreter for the remainder of the current active session. This is a powerful feature, which allows build-as-you-go for a wide range of tasks.

Example:

```
How can I assist you? > Can you write a function which reads an Excel file?
```

-> *model writes a function called read_excel*

```
python > excel_data = read_excel("path/to/excel")
```

## MultiModal Support

In Owlsight 2, models are supported that require additional input, like images or audio. In the backend, this is made possible with the **MultiModalProcessorTransformers** class. 
In the CLI, this can be done by setting the *config.model.model_id* to a multimodal model from the Huggingface modelhub. The model should be a Pytorch model. 
For convenience, it is recommended to select a model through the new Huggingface API in the configuration-settings (read below for more information).

The following tasks are supported for multimodal models:

- image-to-text
- automatic-speech-recognition
- visual-question-answering
- document-question-answering

These models require additional input, which can be passed in the prompt. 
The syntax for passing mediatypes can be done through special double-square brackets syntax, like so:

*How can I assist you?*
```text
[[image:path/to/file.jpg]]
```

The supported mediatypes are: *image*, *audio*.
For example, to pass an image to a document-question-answering model, you can use the following syntax:

*How can I assist you?*
```text
What is the first sentence in this image? [[image:path/to/image.jpg]]
```

## Python interpreter

Next to the fact that objects generated by model-generated code can be accessed, the Python interpreter also has some useful default functions, starting with the "owl_" suffix. These serve as utilityfunctions.

These are:
- `owl_import`: Import Python file to current namespace
- `owl_read`: Read file content from any supported format
- `owl_scrape`: Scrape urls
- `owl_show`: Display active objects in the Python namespace
- `owl_write`: Write content to text file
- `owl_history`: Display model chat history
- `owl_models`: Display loaded HuggingFace models in cache directory
- `owl_press`: Press keys for automation tasks
- `owl_save_namespace`: Save namespace to .dill file
- `owl_load_namespace`: Load namespace from .dill file
- `owl_tools`: Show available functions for tool calling
- `owl_search`: Search and get results from the web using DuckDuckGo's API

## Configurations

Owlsight uses a configuration file in JSON-format to adjust various parameters. The configuration is divided into five main sections: `main`, `model`,  `generate`, `rag` and `huggingface`. Here's an overview of the application architecture:

Main Menu:
- assistant: Chat with the loaded model. Use {{expression}} to pass python code directly. Or e.g. [[image: path/to/image.jpg]] to pass an image to the model
- shell: Execute shell commands
- python: Enter Python interpreter
- config: Configuration settings
  - main settings:
    - back: Return to previous menu
    - max_retries_on_error: Maximum number of retries for Python code error recovery. This parameter is only used when `prompt_retry_on_error` is set to True., Options: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, Type: OptionType.TOGGLE
    - prompt_retry_on_error: Whether to prompt before retrying on error. Set this to True to avoid direct Python code execution on error!, Options: False, True, Type: OptionType.TOGGLE
    - prompt_code_execution: Whether to prompt before executing code. Set this to True to avoid direct Python code execution!, Options: False, True, Type: OptionType.TOGGLE
    - track_model_usage: Show metrics after a model response. Tracks GPU/CPU usage, amount of generated words and responsetime of model. NOTE: GPU tracking only works for PyTorch models., Options: False, True, Type: OptionType.TOGGLE
    - extra_index_url: Additional URL for Python package installation. Useful for example when installing python packages (through pip) from private repositories, Type: OptionType.EDITABLE
    - python_compile_mode: Compile mode in the Python Interpreter (main menu): 'exec' is suited for defining code blocks, 'single' for direct execution, Options: exec, single, Type: OptionType.TOGGLE
    - dynamic_system_prompt: Experimental feature: The model will first act as Prompt Engineer to create a new system prompt based on user input., Options: False, True, Type: OptionType.TOGGLE
    - default_config_on_startup: Link to a configuration file that will be loaded on startup., Type: OptionType.EDITABLE
    - sequence_on_loading: A list of key sequences to execute when loading the configuration. Uses owl_press functionality., Type: OptionType.EDITABLE
  - model settings:
    - back: Return to previous menu
    - model_id: Model identifier or path. The most important parameter in the configuration, as this will load the model to be used, Type: OptionType.EDITABLE
    - apply_chat_history: Toggle the inclusion of saved chat history in the prompt. Enable for chat models, disable for instruct models., Options: False, True, Type: OptionType.TOGGLE
    - system_prompt: System prompt defining model behavior, Type: OptionType.EDITABLE
    - model_kwargs: Additional parameters passed during model initialization. For llama-cpp, these get passed to llama_cpp.Llama. For transformers, these get passed to transformers.pipeline, Type: OptionType.EDITABLE
    - transformers__device: Device for transformers model, Options: None, cpu, cuda, mps, Type: OptionType.TOGGLE
    - transformers__quantization_bits: Quantization bits for transformers model, Options: None, 4, 8, 16, Type: OptionType.TOGGLE
    - transformers__stream: Whether to stream input to transformers model, Options: False, True, Type: OptionType.TOGGLE
    - gguf__filename: GGUF model filename, Type: OptionType.EDITABLE
    - gguf__verbose: Verbose output for GGUF model, Options: False, True, Type: OptionType.TOGGLE
    - gguf__n_ctx: Context length for GGUF model, Options: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, Type: OptionType.TOGGLE
    - gguf__n_gpu_layers: Number of layers from the model which are offloaded to the GPU, Options: -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, Type: OptionType.TOGGLE
    - gguf__n_batch: Batch size to be used by GGUF model, Options: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, Type: OptionType.TOGGLE
    - gguf__n_cpu_threads: Number of CPU threads to be used by GGUF model., Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, Type: OptionType.TOGGLE
    - onnx__model_dir: Directory containing local ONNX model, Type: OptionType.EDITABLE
    - onnx__verbose: Verbose output for ONNX model, Options: False, True, Type: OptionType.TOGGLE
    - onnx__n_cpu_threads: Number of CPU threads to be used by ONNX model, Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, Type: OptionType.TOGGLE
  - generate settings:
    - back: Return to previous menu
    - stopwords: Stopwords that stop text generation. This can be useful for getting more control over when modelgeneration should stop. Pass these like `['stop', 'word']`, Type: OptionType.EDITABLE
    - max_new_tokens: Maximum amount of tokens to generate, Options: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, Type: OptionType.TOGGLE
    - temperature: Temperature for model generation, Options: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, Type: OptionType.TOGGLE
    - generation_kwargs: Additional generation parameters, like top_k, top_p, etc. Pass these like `{'top_k': 4, 'top_p': 0.9}`, Type: OptionType.EDITABLE
  - rag settings:
    - back: Return to previous menu
    - active: Whether RAG for python libraries is active. If True, the search-results will be implicitly added as context to the modelprompt and when pressing ENTER, search-results will be shown, Options: False, True, Type: OptionType.TOGGLE
    - target_library: Target python library for to use for RAG. If the library is not installed in the active environment, a warning will be showed with available options, Type: OptionType.EDITABLE
    - top_k: Number of most matching RAG results to return, based on `search` query, Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, Type: OptionType.TOGGLE
    - sentence_transformer_weight: Weight for the embedding model. TFIDF-weight is 1 - `sentence_transformer_weight`, Options: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, Type: OptionType.TOGGLE
    - sentence_transformer_name_or_path: Name or path to a sentence-transformer model, which is used for embedding, Type: OptionType.EDITABLE
    - search: RAG search query. Press ENTER to show the `top_k` results. Only used when `active` is True, Type: OptionType.EDITABLE
  - agentic settings:
    - back: Return to previous menu
    - apply_tools: Toggle whether the agentic system is active. Available tools concerns an existing subset of functions (and every new defined one) in the Python Interpreter namespace., Options: False, True, Type: OptionType.TOGGLE
    - max_steps: Maximum number of steps for the agentic system., Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, Type: OptionType.TOGGLE
    - enable_python_agent: Toggle the inclusion of a Python generation agent. This agent judges the last response of the Tool agent and writes Python code if appropriate, Options: False, True, Type: OptionType.TOGGLE
  - huggingface settings:
    - back: Return to previous menu
    - search: Search for a model on the Hugging Face Hub by pressing ENTER. Keywords can be used optionally to finetune searchresults, e.g. 'llama 3b gguf', Type: OptionType.EDITABLE
    - top_k: Top number of Hugging Face results to return. The results will be sorted by highest score first, Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, Type: OptionType.TOGGLE
    - select_model: Select and load a model from the Hugging Face Hub by toggling through the options found by `search`, Type: OptionType.TOGGLE
    - task: Filter Hugging Face models by task. When using `search`, the results will be filtered directly by chosen task, Options: None, text-generation, text2text-generation, translation, summarization, image-to-text, automatic-speech-recognition, visual-question-answering, document-question-answering, Type: OptionType.TOGGLE
- save: Save current configuration as JSON-file
- load: Load a configuration from a JSON-file
- clear history: Clear owlsight cache (directory called '.owlsight') and chat history
- quit: Exit application

Here's an example of what the default configuration looks like:

```json
{
    "main": {
        "max_retries_on_error": 3,
        "prompt_retry_on_error": true,
        "prompt_code_execution": true,
        "track_model_usage": false,
        "extra_index_url": "",
        "python_compile_mode": "single",
        "dynamic_system_prompt": false,
        "default_config_on_startup": "",
        "sequence_on_loading": []
    },
    "model": {
        "model_id": "",
        "apply_chat_history": true,
        "system_prompt": "",
        "model_kwargs": {},
        "transformers__device": null,
        "transformers__quantization_bits": null,
        "transformers__stream": true,
        "gguf__filename": "",
        "gguf__verbose": false,
        "gguf__n_ctx": 2048,
        "gguf__n_gpu_layers": 0,
        "gguf__n_batch": 8,
        "gguf__n_cpu_threads": 8,
        "onnx__model_dir": "",
        "onnx__verbose": false,
        "onnx__n_cpu_threads": 8
    },
    "generate": {
        "stopwords": [],
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "generation_kwargs": {}
    },
    "rag": {
        "active": false,
        "target_library": "",
        "top_k": 10,
        "sentence_transformer_weight": 0.0,
        "sentence_transformer_name_or_path": "Alibaba-NLP/gte-base-en-v1.5",
        "search": ""
    },
    "agentic": {
        "apply_tools": false,
        "max_steps": 5,
        "enable_python_agent": true
    },
    "huggingface": {
        "search": "",
        "top_k": 10,
        "select_model": "",
        "task": null
    }
}
```

Configuration files can be saved (`save`) and loaded (`load`) through the main menu.

### Changing configurations

To update a configuration, simply modify the desired value and press **ENTER** to confirm the change. Please note that only one configuration setting can be updated at a time, and the change will only go into effect once **ENTER** has been pressed.

## Temporary environment

During an Owlsight session, a temporary environment is created within the homedirectory, called ".owlsight_packages". Newly installed python packages will be installed here. This folder will be removed if the session ends. If you want to persist installed packages, simply install them outside of Owlsight.

## Error Handling and Auto-Fix

Owlsight automatically tries to fix and retry any code that encounters a **ModuleNotFoundError** by installing the required package and re-executing the code. It can also attempt to fix errors in its own generated code. This feature can be controlled by the *max_retries_on_error* parameter in the configuration file.

## API Examples

Owlsight can also be used as a library in Python scripts. The main classes are the `TextGenerationProcessor` family, which can be imported from the `owlsight` package. 

Here is a simple example of how to use it:
```python
from owlsight import TextGenerationProcessorGGUF
# If you want to use another type of text-generation model, you can import the other classes: TextGenerationProcessorONNX, TextGenerationProcessorTransformers

processor = TextGenerationProcessorGGUF(
    model_id=r"path	o\Phi-3-mini-128k-instruct.Q5_K_S.gguf",
)

question = "What is the meaning of life?"

for token in processor.generate_stream(question):
    print(token, end="", flush=True)
```

Alternatively, there is a lot more to explore in the `owlsight` package.
Here is an example on how to use the `DocumentSearcher` class for simple document retrieval:
```python
from owlsight import DocumentSearcher, SentenceTextSplitter, SemanticTextSplitter

docs = {
    "doc1": "Quantum mechanics describes nature at atomic scales, introducing wave-particle duality and entanglement.",
    "doc2": "General relativity redefines gravity as spacetime curvature, predicting black holes and gravitational waves.",
    "doc3": "Quantum gravity aims to unify quantum mechanics and relativity, with theories like string theory and LQG.",
    "doc4": "String theory is a framework for understanding the universe, with models like the Minkowski space-time and the Einstein-Hilbert action."
    "doc5": "LQG is a framework for quantum gravity, with models like the Einstein action and the black hole metric."
}

# Experiment with different text splitters
# splitter = SemanticTextSplitter()
splitter = SentenceTextSplitter(n_sentences=2)

searcher = DocumentSearcher(
    documents=docs,
    text_splitter=splitter,
    cache_dir="quantum_gravity",
    cache_dir_suffix="test",
)

query = "black holes in quantum gravity"
results = searcher.search(query, top_k=2)
```


## API Documentation

The following section details all the objects and functions available in the Owlsight API:


### Classes

#### TextGenerationProcessorOnnx

```python
class TextGenerationProcessorOnnx(model_id: str, onnx__verbose: bool = False, onnx__n_cpu_threads: int = 8, onnx__model_dir: Optional[str] = None, token: Optional[str] = None, apply_chat_history: bool = False, system_prompt: Optional[str] = None, model_kwargs: Optional[dict] = None, apply_tools: Optional[List[dict]] = None, **kwargs: Any) -> None
```

Text generation processor using ONNX Runtime optimized models.

This processor enables text generation using ONNX-optimized models,
which can run on both CPU and GPU. Supports both local models and models from
Hugging Face Hub.

Parameters
----------
model_id : str
    Path to local ONNX model or Hugging Face model ID
onnx__verbose : bool, default=False
    Enable verbose ONNX Runtime logging
onnx__n_cpu_threads : int, default=8
    Number of CPU threads for computation
onnx__model_dir : str, optional
    Specific model directory when multiple valid ones exist
token : str, optional
    Hugging Face token for private models
apply_chat_history : bool, default=False
    Whether to maintain conversation history
system_prompt : str, optional
    System prompt prepended to all inputs
model_kwargs : dict, optional
    Additional keyword arguments to pass to the model.
    Default is None.

Notes
-----
- ONNX models typically offer better CPU performance than PyTorch
- Thread count affects CPU performance significantly
- Models must be ONNX-optimized versions of transformers models

**Examples:**

```python
--------
>>> # Load local ONNX model
>>> processor = TextGenerationProcessorOnnx("path/to/model")
>>>
>>> # Load from Hugging Face
>>> processor = TextGenerationProcessorOnnx(
...     "onnx-community/Llama-2-7B-Instruct-ONNX",
...     onnx__n_cpu_threads=12
... )
```

**Methods:**

- `apply_chat_template(self, input_data: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer) -> str`
  - Apply chat template to the input text.
- `generate(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, stopwords: Optional[List[str]] = None, buffer_wordsize: int = 10, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text response for the given input.
- `generate_stream(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, generation_kwargs: Optional[Dict[str, Any]] = None)`
  - Stream generated text tokens one by one.
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self) -> Optional[int]`
  - Get maximum context length for the model.
- `list_valid_repo_files(repo_id: str) -> List[str]`
- `pre_validate_model_id(model_id: str, onnx__model_dir: str)`
  - Validate the model_id and model_directory before using `snapshot_download`.
- `update_history(self, input_data: str, generated_text: str) -> None`
  - Update the history with the input and generated text.

#### TextGenerationProcessorTransformers

```python
class TextGenerationProcessorTransformers(model_id: str, transformers__device: Optional[str] = None, transformers__quantization_bits: Optional[int] = None, transformers__stream: bool = True, bnb_kwargs: Optional[dict] = None, tokenizer_kwargs: Optional[dict] = None, task: Optional[str] = None, apply_chat_history: bool = False, system_prompt: str = '', apply_tools: Optional[List[dict]] = None, model_kwargs: Optional[dict] = None, **kwargs)
```

Text generation processor using transformers library.

**Methods:**

- `apply_chat_template(self, input_data: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer) -> str`
  - Apply chat template to the input text.
- `generate(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, stopwords: Optional[List[str]] = None, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text response.
- `generate_stream(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, stopwords: Optional[List[str]] = None, generation_kwargs: Optional[Dict[str, Any]] = None) -> Generator[str, NoneType, NoneType]`
  - Generate streaming text response.
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self) -> Optional[int]`
  - Retrieve the maximum context length of the model.
- `pipe_call(self, input_data: Union[str, List[str]], **gen_kwargs) -> Any`
  - Call the pipeline with input data and kwargs, supporting batch processing.
- `prepare_generation(self, input_data: str, max_new_tokens: int, temperature: float, stopwords: Optional[List[str]], generation_kwargs: Optional[Dict[str, Any]], streaming: bool = False, apply_chat_template: bool = True) -> Tuple[str, Dict[str, Any]]`
  - Prepare generation parameters.
- `update_history(self, input_data: str, generated_text: str) -> None`
  - Update the history with the input and generated text.

#### TextGenerationProcessorGGUF

```python
class TextGenerationProcessorGGUF(model_id: str, gguf__filename: str = '', gguf__verbose: bool = False, gguf__n_ctx: Optional[int] = None, gguf__n_gpu_layers: int = 0, gguf__n_batch: Optional[int] = None, gguf__n_cpu_threads: Optional[int] = None, apply_chat_history: bool = False, system_prompt: str = '', model_kwargs: Dict[str, Any] = None, apply_tools: Optional[List[dict]] = None, **kwargs)
```

Text generation processor for GGUF models using llama-cpp.

This processor enables efficient text generation using GGUF-quantized models,
which are optimized for CPU and GPU inference. Supports both local models and
models from Hugging Face Hub.

Parameters
----------
model_id : str
    Path to local GGUF model or Hugging Face model ID
gguf__filename : str, optional
    Specific GGUF file to load when using Hugging Face model ID
gguf__verbose : bool, default=False
    Enable verbose logging from llama-cpp
gguf__n_ctx : int, optional
    Context window size. Larger values allow longer conversations but use more memory
gguf__n_gpu_layers : int, default=0
    Number of layers to offload to GPU. Set >0 for GPU acceleration
gguf__n_batch : int, optional
    Batch size for generation. Increase for faster generation, at the cost of memory.
gguf__n_cpu_threads : int, optional
    The number of CPU threads to use for generation. Increase for much faster generation if multiple cores are available.
apply_chat_history : bool, default=False
    Whether to maintain conversation history
system_prompt : str, default=""
    System prompt prepended to all inputs
model_kwargs : Optional[Dict[str, Any]]
    Additional arguments passed for the model.
    These get passed to `transformers.pipeline` function as `model_kwargs` argument.
    Default is None.

Notes
-----
- GPU acceleration requires llama-cpp-python build specifically with CUDA support
- Context size (n_ctx) affects memory usage significantly
- For optimal performance, adjust n_batch and n_cpu_threads based on hardware

**Examples:**

```python
--------
>>> # Load local GGUF model
>>> processor = TextGenerationProcessorGGUF("path/to/model.gguf", gguf__n_gpu_layers=20)
>>>
>>> # Load from Hugging Face with GPU
>>> processor = TextGenerationProcessorGGUF(
...     "TheBloke/Llama-2-7B-GGUF",
...     gguf__filename="llama-2-7b.Q4_K_M.gguf",
...     gguf__n_gpu_layers=32
... )
```

**Methods:**

- `apply_chat_template(self, input_data: str) -> List[Dict[str, str]]`
  - Apply chat template to the input text.
- `generate(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.1, stopwords: Optional[List[str]] = None, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text response for the given input.
- `generate_stream(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.1, generation_kwargs: Optional[Dict[str, Any]] = None) -> Generator[str, NoneType, NoneType]`
  - Stream generated text tokens one by one.
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self) -> Optional[int]`
  - Retrieve the maximum context length of the model.
- `update_history(self, input_data: str, generated_text: str) -> None`
  - Update the history with the input and generated text.

#### MultiModalProcessorTransformers

```python
class MultiModalProcessorTransformers(model_id: str, task: str, apply_chat_history: bool = False, system_prompt: str = '', **kwargs: Any) -> None
```

Multimodal processor using Hugging Face transformers.

This processor handles text generation tasks that involve multiple modalities
(text, images, audio) using Hugging Face transformer models. It combines
the MediaPreprocessor for handling media inputs with text generation capabilities.

Parameters
----------
model_id : str
    Identifier for the Hugging Face model to use
task : str
    Task type, must be one of HUGGINGFACE_MEDIA_TASKS
apply_chat_history : bool, default=False
    Whether to maintain chat history
system_prompt : str, default=""
    System prompt to use for generation
**kwargs : dict
    Additional arguments passed to TextGenerationProcessorTransformers

Notes
-----
- Supports various multimodal tasks (VQA, image captioning, etc.)
- Handles media preprocessing automatically
- Integrates with Hugging Face's transformers library
- Manages memory efficiently for large media files

**Examples:**

```python
--------
>>> processor = MultiModalProcessorTransformers(
...     model_id="dandelin/vilt-b32-finetuned-vqa",
...     task="visual-question-answering"
... )
>>> media_obj = MediaObject(path="image-of-car.jpg", tag="image")
>>> result = processor.generate(
...     "What color is the car in this image:",
...     media_objects={"image1": media_obj}
... )
```

**Methods:**

- `apply_chat_template(self, input_data: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer) -> str`
  - Apply chat template to the input text.
- `generate(self, input_data: str, media_objects: Dict[str, owlsight.utils.custom_classes.MediaObject], stopwords: Optional[List[str]] = None, max_new_tokens: int = 512, temperature: float = 0.0, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text based on input text and media objects.
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self)`
  - Retrieve the maximum context length of the model.
- `preprocess_input(self, input_data: Union[str, bytes, pathlib.Path], question: Optional[str] = None) -> Any`
  - Preprocess media input data for the model.
- `update_history(self, input_data: str, generated_text: str) -> None`
  - Update the history with the input and generated text.

#### PythonLibSearcher

```python
class PythonLibSearcher(*args, **kwargs)
```

A singleton class for searching Python library documentation with caching capabilities.
Maintains document and engine caches throughout the owlsight session.

**Methods:**

- `clear_cache(self, library: Optional[str] = None)`
  - Clear the document and engine caches.
- `search(self, library: str, query: str, top_k: int = 5, cache_dir: Optional[str] = None, as_context: bool = True, tfidf_weight: float = 1.0, sentence_transformer_weight: float = 0.0, sentence_transformer_model: str = 'Alibaba-NLP/gte-base-en-v1.5') -> Union[pandas.core.frame.DataFrame, str]`
  - Search Python library documentation with caching for documents and search engines.

#### SentenceTextSplitter

```python
class SentenceTextSplitter(n_sentences: int = 3, n_overlap: int = 0)
```

Split text into chunks based on sentences.

**Methods:**

- `split_and_clean_text(text: str) -> List[str]`
  - Split a longer text into sentences and clean them.
- `split_documents(self, documents: Dict[str, str], **kwargs) -> Dict[str, str]`
  - Split documents into chunks of n sentences with overlap.
- `split_text_in_sentences(text: str) -> List[str]`
  - Split a longer text into sentences, while keeping account edgecases.

#### SemanticTextSplitter

```python
class SemanticTextSplitter(model_name: Optional[str] = 'Alibaba-NLP/gte-base-en-v1.5', window_size: int = 1, percentile: float = 0.9, device: Optional[str] = None)
```

Split text into chunks based on semantic similarity breakpoints.

**Methods:**

- `set_model(self, model: Union[str, ~SentenceTransformer]) -> None`
  - Set the model for the splitter.
- `split_documents(self, documents: Dict[str, str], show_progress_bar: bool = True, **kwargs) -> Dict[str, str]`
  - Split documents using semantic breakpoint detection.

#### DocumentSearcher

```python
class DocumentSearcher(documents: Dict[str, str], sentence_transformer_model: str = 'Alibaba-NLP/gte-base-en-v1.5', sentence_transformer_batch_size: int = 64, text_splitter: Optional[owlsight.rag.text_splitters.TextSplitter] = None, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None) -> None
```

Document search engine using an ensemble of TFIDF and Sentence Transformer methods.

This class provides document search capability by combining traditional TF-IDF
with modern neural embeddings. The idea behind this is two-fold:
- TFIDF can capture relevant words an embedding model was not trained on.
- Embeddings can capture context better than TFIDF.

Order in __init__is like so:
[splitting in chunks (optional)]
[TF-IDF]
[Sentence Transformer: create embeddings and cache as .pkl files]

And then use the `search` method to combine the results:
[Combine TF-IDF and Sentence Transformer results]

Parameters
----------
documents : Dict[str, str]
    Dictionary mapping document IDs to their content
sentence_transformer_model : str, default='Alibaba-NLP/gte-base-en-v1.5'
    Name or path of the Sentence Transformer model
sentence_transformer_batch_size : int, default=64
    Batch size for computing embeddings
text_splitter : Optional[TextSplitter], default=None
    Strategy for splitting documents into chunks. If None, no splitting is done.
cache_dir : str, optional
    Directory to cache embeddings and results
cache_dir_suffix : str, optional
    Suffix for cache directory name.
    Recommended is to use a name which is unique and descriptive to the documents.

Notes
-----
- Uses both TF-IDF and neural embeddings for robust search
- Has caching capabilities in pickled files
- Supports batch processing for efficient embedding computation
- Supports custom text splitting strategies through the TextSplitter interface

**Examples:**

```python
--------
>>> docs = {
...     "doc1": "Python is a programming language",
...     "doc2": "Machine learning is fascinating"
... }
>>> # Using default sentence-based splitting
>>> splitter = SentenceTextSplitter(n_sentences=3, n_overlap=1)
>>> searcher = DocumentSearcher(docs, text_splitter=splitter, cache_dir="document_cache", cache_dir_suffix="programming")
>>> results = searcher.search("python programming", top_k=3)
```

**Methods:**

- `search(self, query: str, top_k: int = 20, sentence_transformer_weight: float = 0.7, tfidf_weight: float = 0.3, as_context: bool = False) -> Union[pandas.core.frame.DataFrame, str]`
  - Search documents using the configured ensemble methods.

#### DocumentReader

```python
class DocumentReader(supported_extensions: Optional[List[str]] = None, ignore_patterns: Optional[List[str]] = None, ocr_enabled: bool = True, timeout: int = 5, text_only: bool = True, tika_server_jar_path: Optional[str] = None)
```

A class for reading text content from files using Apache Tika.

Supports a wide variety of file formats and provides streaming capabilities
for processing large directories.

**Examples:**

```python
--------
>>> reader = DocumentReader()
>>> for filename, content in reader.read_directory("path/to/docs"):
...     print(f"Processing {filename}...")
...     process_content(content)
```

**Methods:**

- `is_supported_file(self, filepath: str) -> bool`
  - Check if a file is supported based on its extension and ignore patterns.
- `read_directory(self, directory: str, recursive: bool = True) -> Generator[Tuple[str, str], NoneType, NoneType]`
  - Read all supported files in a directory and yield their content.
- `read_file(self, filepath: str) -> Optional[str]`
  - Read and extract text content from a single file.
- `should_ignore_file(self, filepath: str) -> bool`
  - Check if a file should be ignored based on gitignore-style patterns.

#### HashingVectorizerSearchEngine

```python
class HashingVectorizerSearchEngine(documents: Dict[str, str], cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, **hashing_kwargs: Any)
```

Search engine using Hashing Vectorizer for memory-efficient search.

This search engine uses feature hashing for vectorization, making it memory-efficient
and suitable for large document collections.

Parameters
----------
documents : Dict[str, str]
    Dictionary mapping document IDs to their content
cache_dir : str, optional
    Directory to cache hash matrices
cache_dir_suffix : str, optional
    Suffix for cache directory name
**hashing_kwargs
    Additional arguments passed to sklearn.feature_extraction.text.HashingVectorizer

Notes
-----
- Memory-efficient, suitable for large datasets
- No inverse transform capability
- Constant memory usage regardless of vocabulary size
- Small chance of hash collisions

**Examples:**

```python
--------
>>> docs = {
...     "doc1": "Large text document...",
...     "doc2": "Another large document..."
... }
>>> engine = HashingVectorizerSearchEngine(
...     docs,
...     n_features=(2**16)
... )
>>> results = engine.search("specific terms", top_k=1)
```

**Methods:**

- `create_index(self) -> None`
  - Create search index from documents.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Search documents using the query.

#### TFIDFSearchEngine

```python
class TFIDFSearchEngine(documents: Dict[str, str], cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, **tfidf_kwargs: Any) -> None
```

Search engine using TF-IDF (Term Frequency-Inverse Document Frequency).

This search engine uses traditional TF-IDF vectorization for keyword-based search,
making it effective for finding documents with specific terms.

Parameters
----------
documents : Dict[str, str]
    Dictionary mapping document IDs to their content
cache_dir : str, optional
    Directory to cache TF-IDF matrices
cache_dir_suffix : str, optional
    Suffix for cache directory name
**tfidf_kwargs
    Additional arguments passed to sklearn.feature_extraction.text.TfidfVectorizer

Notes
-----
- Fast and memory-efficient
- Good for exact keyword matching
- Supports n-grams and custom tokenization
- Caches TF-IDF matrices for better performance

**Examples:**

```python
--------
>>> docs = {
...     "doc1": "Python programming basics",
...     "doc2": "Advanced Python concepts"
... }
>>> engine = TFIDFSearchEngine(docs, ngram_range=(1, 2))
>>> results = engine.search("python basics", top_k=1)
```

**Methods:**

- `create_index(self) -> None`
  - Create search index from documents.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Search documents using the query.

#### SentenceTransformerSearchEngine

```python
class SentenceTransformerSearchEngine(documents: Dict[str, str], model_name: str = 'Alibaba-NLP/gte-base-en-v1.5', pooling_strategy: Literal['mean', 'max', None] = 'mean', device: Optional[str] = None, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, batch_size: int = 64)
```

Search engine using Sentence Transformer embeddings.

This search engine uses neural embeddings to find semantically similar documents,
making it effective for concept-based search rather than just keyword matching.

Parameters
----------
documents : Dict[str, str]
    Dictionary mapping document IDs to their content
model_name : str, default='Alibaba-NLP/gte-base-en-v1.5'
    Name or path of the Sentence Transformer model
pooling_strategy : {'mean', 'max', None}, default='mean'
    Strategy for pooling sentence embeddings:
    - 'mean': Average embeddings (better for context)
    - 'max': Maximum values (better for key concepts)
    - None: No pooling (for single-sentence documents)
device : str, optional
    Device to run model on ('cpu', 'cuda', etc.)
cache_dir : str, optional
    Directory to cache embeddings
cache_dir_suffix : str, optional
    Suffix for cache directory name
batch_size : int, default=64
    Batch size for computing embeddings

Notes
-----
- Provides semantic search capability
- Automatically handles sentence splitting and pooling
- Supports GPU acceleration
- Caches embeddings for better performance

**Examples:**

```python
--------
>>> docs = {
...     "doc1": "Python is great for machine learning",
...     "doc2": "Deep learning revolutionized AI"
... }
>>> engine = SentenceTransformerSearchEngine(
...     docs,
...     model_name='all-MiniLM-L6-v2',
...     pooling_strategy='mean'
... )
>>> results = engine.search("AI and ML", top_k=1)
```

**Methods:**

- `create_index(self) -> None`
  - Create search index by computing embeddings for all documents.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Search documents using the query.

#### OwlDefaultFunctions

```python
class OwlDefaultFunctions(globals_dict: owlsight.utils.custom_classes.GlobalPythonVarsDict)
```

Define default functions that can be used in the Python interpreter.
This provides the user with some utility functions to interact with the interpreter.
Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

This class is open for extension, as possibly more useful functions can be added in the future.

**Methods:**

- `owl_import(self, file_path: str)`
  - Import Python module into the current execution environment.
- `owl_load_namespace(self, file_path: str)`
  - Load namespace using dill.
- `owl_models(self, cache_dir: Optional[str] = None, show_task: bool = False) -> List[str]`
  - Audit Hugging Face model cache.
- `owl_press(self, sequence: List[str], exit_python_before_sequence: bool = True, time_before_sequence: float = 0.5, time_between_keys: float = 0.12) -> bool`
  - Simulate keyboard input for application control.
- `owl_read(self, path: Union[str, pathlib.Path, Iterable[Union[str, pathlib.Path]]], recursive: bool = False, ignore_patterns: Optional[List[str]] = None, ocr_enabled: bool = True, timeout: int = 5) -> Union[str, Dict[str, str]]`
  - Read LOCAL FILE CONTENTS with advanced document processing.
- `owl_save_namespace(self, file_path: str)`
  - Serialize current namespace state to disk.
- `owl_scrape(self, urls: List[str], max_concurrent: int = 5) -> str`
  - Scrape web content from URLs (use instead of owl_read for web resources).
- `owl_search(self, query: str, max_results: int = 10, max_retries: int = 3) -> list`
  - Execute web search using DuckDuckGo's API.
- `owl_show(self, docs: bool = True, return_str: bool = False) -> List[str]`
  - Display active namespace objects with documentation.
- `owl_tools(self, as_json: bool = True) -> List[Union[Callable, Dict]]`
  - Retrieve available tool-callable functions in OpenAI-compatible format.
- `owl_write(self, file_path: str, content: str) -> None`
  - Write text content to filesystem.

#### ExpertPrompts

```python
class ExpertPrompts()
```

System prompts for different expert roles

**Methods:**

- `as_dict(self) -> Dict[str, str]`
  - Return a dictionary of role keys and their descriptions.

#### AgentPrompts

```python
class AgentPrompts(essential_information: str = '')
```

A collection of system prompts to be used in Agentic frameworks.

**Methods:**

- `as_dict(self) -> Dict[str, str]`
  - Return a dictionary of role keys and their descriptions.
- `get_essential_information(self) -> str`
- `get_single_agent(self) -> str`

#### PromptWriter

```python
class PromptWriter(prompt: str)
```

Writes a system prompt to an Owlsight configuration JSON file.

Parameters
----------
prompt : str
    The system prompt to be written to the Owlsight configuration JSON file.

**Methods:**

- `to(self, target_json: str) -> None`
  - Updates the 'system_prompt' field under the 'model' key in the given Owlsight configuration JSON file.


### Functions

#### setup_tesseract

```python
def setup_tesseract() -> str
```

Initialize Tesseract. Return the path to the Tesseract executable.

#### get_best_device

```python
def get_best_device() -> str
```

Check for best device and return the device name.

#### check_onnx_device

```python
def check_onnx_device(current_device: str = 'cuda') -> str
```

Check the current device being used for ONNXRuntime.

Parameters:
current_device (str): The current device to use. Default is 'cuda'.

#### check_gpu_and_cuda

```python
def check_gpu_and_cuda()
```

Checks if a CUDA-capable GPU is available on pytorch and if CUDA is installed.

#### llama_supports_gpu_offload

```python
def llama_supports_gpu_offload(base_path: str) -> bool
```

Checks if Llama.cpp supports GPU offload. This is useful for checking if a GPU is available for GGUF models.

Parameters
----------
base_path : str
    Path to the Llama.cpp shared library.
    Usually something like 'dist-packages/llama_cpp/lib' or 'site-packages/llama_cpp/lib'
    in the current virtual environment.

Returns:
    bool: True if Llama.cpp is available on the GPU, False otherwise.

SOURCE: https://stackoverflow.com/questions/78415856/detecting-gpu-availability-in-llama-cpp-python

#### calculate_max_parameters_per_dtype

```python
def calculate_max_parameters_per_dtype()
```

Calculate the maximum number of parameters that can be run on the GPU
for different data types (32-bit, 16-bit, 8-bit, 4-bit).

#### calculate_memory_for_model

```python
def calculate_memory_for_model(n_bilion_parameters: int, n_bit: int = 32) -> float
```

Calculate the memory required for a model in GB.

Parameters:
n_bilion_parameters (int): The number of parameters in the model in billions.
n_bit (int): The number of bits used to represent the model parameters. Default is 32. Quantized models use 16/8/4 bits.

#### calculate_available_vram

```python
def calculate_available_vram() -> float
```

Calculate the available VRAM on the GPU in GB.

#### select_processor_type

```python
def select_processor_type(model_id: str, task: Optional[str] = None) -> Type[ForwardRef('TextGenerationProcessor')]
```

Utilityfunction which selects the appropriate TextGenerationProcessor class based on the model ID or directory.

If the model_id is a directory, the function will inspect the contents of the directory
to decide the processor type. Otherwise, it will use the model_id string to make the decision.

#### is_url

```python
def is_url(url: str) -> bool
```

Check if a string is a valid URL.

Parameters
----------
url : str
    The string to check.

Returns
-------
bool
    True if the string is a valid URL, False otherwise.

#### get_model_data

```python
def get_model_data(model_search: str, top_n_models: int = 10, **kwargs) -> Dict[str, Dict[str, str]]
```

Get and display the model data from the HuggingFace Hub in a visually appealing format.

Parameters:
    model_search: Search term for filtering models
    top_n_models: Number of top models to display
    **kwargs: Additional keyword arguments to pass to get_model_list. E.g., task, framework, etc.
    See `HfApi().list_models()` from `huggingface_hub` package for more details.

Returns:
    Dictionary containing model information

#### function_to_json_for_tool_calling

```python
def function_to_json_for_tool_calling(func: Callable) -> dict
```

Converts a Python function into a JSON structure suitable for function-calling
with an LLM. This function inspects the target function's signature and docstring
(assumed to be in NumPy style) and returns a JSON schema-like definition.

Parameters
----------
func : Callable
    The Python function to be converted.

Returns
-------
str
    A JSON string describing the function's name, short description, and parameter schema.



## RELEASE NOTES

**1.0.2**

- Enhanced cross-platform compatibility.
- Introduced the `generate_stream` method to all `TextGenerationProcessor` classes.
- Various minor bug fixes.

**1.1.0**

- Added Retrieval Augmented Generation (RAG) for enriching prompts with documentation from python libraries. This option is also added to the configuration.
- History with autocompletion is now also available when writing prompts. Prompts can be autocompleted with TAB.

**1.2.1**

- Access backend functionality through the Owlsight API using "from owlsight import ..."
- Added default functions to the Python interpreter, starting with the "owl_" suffix.
- More configurations available when using GGUF models from the command line.

**1.3.0**

- Add `owl_history` function to python interpreter for directly accessing model chat history.
- Improved validation when  loading a configuration file.
- Added validation for retrying a codeblock from an error. This configuration is called `prompt_retry_on_error`

**1.4.1**

- improve RAG capabilities in the Owlsight API, added **SentenceTransformerSearchEngine**, **TFIDFSearchEngine** and **HashingVectorizerSearchEngine** as classes.
- Added **DocumentSearcher** to offer a general RAG solution for documents. At its core, uses a combination of TFIDF and Sentence Transformer.
- Added caching possibility to all RAG solutions in the Owlsight API (*cache_dir* & *cache_dir_suffix*), where documents, embeddings etc. get pickled. This can save a big amount of time if amount of documents is large.

**2.0.1beta**

*BREAKING CHANGES*

- Added Huggingface API in the configuration-settings of the CLI. This allows the user to search and load models directly from the Huggingface modelhub and can be found through `config:huggingface`.
- added `transformers__use_fp16` and `transformers__stream` to `config:model` for using fp16 and streaming the model output in the transformers-based models.
- Added **MultiModalProcessorTransformers** for non text-input based models. This class can be used for models which require additional input like images, audio or video and works with models from the Huggingface Hub based on the Pytorch framework.
- Introduced new double-square brackets syntax for passing mediatypes in the prompt.
- Improved logging with clearer color coding and more detailed information.
- System Prompt in config:modelis now an empty string as default.
- Several small bugfixes and improvements.

**2.0.2 (stable)**

- Upgraded UI with new color scheme and improved readability. Description of the current choice is now displayed above the menu.
- Removed `onnx__tokenizer` from `TextGenerationProcessorOnnx` constructor, so that only *model_id* is needed as constructor argument.
- Added `get_max_context_length` method to all `TextGenerationProcessor` classes, which returns the maximum context length of the loaded model.
- Moved `transformers__use_fp16` in config:model to `transformers__quantization_bits` as value 16, as it is more clear.
- Added `track_model_usage` to config:main, which can be used to track usage of the model, like the amount of words generated, total time spent etc.
- Added possibility to pass complete directories as argument to mediatypes to a model in the CLI, like so: 

*How can I assist you?*
```text
[[image:directory/containing/images]]
```

- Add `owl_models()` function to python interpreter for displaying all Huggingface models in the cache directory.

**2.2.0**

- Improved userexperience in the CLI by preventing shrinking of the terminal window if menu is too large.
- In the EDITABLE optiontype fields, multiple lines are now possible.
- Add `owl_save_namespace` `owl_load_namespace` functions to save and load all variables inside the Python interpreter. This 
is useful if you want to save any code created by a model. Or load a namespace from a previous session.
- `ProcessorMemoryContext` can be used as a context_manager to clean up resources from `TextGenerationProcessor`, like the model, from memory after usage.
- Improved `config:rag` functionality with the new `sentence_transformer_weight` option. This allows to weigh the sentence-transformer part in the RAG model next to the already present TFIDF, improving semantic search capabilities.
- Improved `config:rag` functionality with the new `sentence_transformer_name_or_path` option. This allows to specify the name or path to a sentence-transformer model, which is used for embedding.
- Add `DocumentSearcher` class to offer a general RAG solution for documents. At its core, uses a combination of TFIDF and Sentence Transformer.
- Add `DocumentReader` class to read text from a broad range of file formats. This class is build on top of Apache Tika.
- Improved `owl_read` with the new `DocumentReader` class. As input, you can now pass a directory or a list of files.
- Added `main:sequence_on_loading` to the configuration json. This allows execution of a sequence of keys on loading a config through the `load` option in the Owlsight main-menu.
TIP: above option can be used to load a sequence of different models as "agents", where every config can be threaded as a different agent with their own role. In theory, every action in Owlsight can be automated through this option.

**2.3.0**

- Added compile mode for the Python interpreter (`config:main:python_compile_mode`), so that the user can both execute single lines ("single") or define multiple lines of code ("exec").
- added `split_documents_n_sentences` and `split_documents_n_overlap` parameters to `DocumentSearcher` class, which can be used to split a long document into smaller chunks before embedding.
- Added a `from_cache` method in DocumentSearcher class. This method can be used to load a DocumentSearcher instance from earlier cached documents and embeddings.
- Removed `transformers__model_kwargs` from config:model, and instead added a `model_kwargs` parameter to all TextGenerationProcessor classes. 
The advantage is that `model_kwargs` can now also be passed to other TextGenerationProcessor classes. For example, when passed to `TextGenerationProcessorGGUF`, these parameters are now used to initialize the `Llama` class from llama-cpp-python.
- ESC + V can be used inside the Python Interpreter to show the currently defined objects in a dropdown-menu.
- ESC + V can be used inside the "How can I assist you?"-option after typing the following: "[[", "{{". This will autocomplete the following:
"[[" will autocomplete to: "image:", "audio:"
"{{" will autocomplete any available defined objects from the python-namespace.
- Added `owl_tools` function to the Python interpreter. This function can be used to convert all defined functions in the namespace to a dictionary, which can be used for tool/function-calling.
- Bracket-syntax "{{}}" for augmenting Python expressions can now also be used inside the `config` section of the CLI. For example, in the Python interpreter, we can store a long string inside a variable and pass it to `config:model:system_prompt` directly.
- Added new option `dynamic_system_prompt` to config:main section. This option can be used to dynamically generate a fitting system prompt first for a given user input, before passing it to the model.
The idea is that this might help the model to give a more focused response to the question.
- Add basic functionality, like select all, copy and paste. Use CTRL+A, CTRL+C and CTRL+Y respectively. This option applies to all editable fields and the Python Interpreter.

**2.4.0** (TO BE RELEASED)

***Several changes for the "How can I assist you?"-option:***
- Added `[[load:...]]` tag support for dynamic configuration loading during conversations. This can be used in "How can I assist you?" in mainmenu to chain multiple configurations (agents) together, like so:

*How can I assist you?*
```text
[[load:config-to-model1.json]] Generates a rough draft for the following text: {{owl_read("mockup-idea.txt")}} [[load:config-to-model2.json]] Validate that the generated draft based on the previous text is relevant and contains all necessary information
```
TIP 1: Combing a sequence of different agents together with above method can lead to complex conversation flows.
TIP 2: Using above tag in combination with `sequence_on_loading` in the configuration json opens lots of new possibilities to control the application.

- Added `[[chain:...]]` tag support for changing config parameters in between conversations. For example: 

*How can I assist you?*
```text
[[chain:model.system_prompt=act as a helpful assistant||generate.temperature=0.5]]
```
- Above tags can also be used INSIDE a python-expression inside the "How can I assist you?"-option, like so:

*How can I assist you?*
```text
{{"".join(f"[[load:config-to-model{i}.json]]how much is {i} + 1?" for i in range(1, 10))}}
```

- Added `SentenceTextSplitter` to the Owlsight API. This can be used to split text into chunks based on sentences.
- Added `SemanticTextSplitter` to the Owlsight API. This can be used to split text into chunks based on semantic similarity breakpoints and might be more accurate for chunking than `SentenceTextSplitter`.
Note that both TextSplitter classes can be used as input for the `DocumentSearcher` class.
- Added `main.default_config_on_startup` to the `config:main` section. This option can be used to specify a default configuration file to load when starting Owlsight.
This will load the configuration file specified in `main.default_config_on_startup` when every time when starting Owlsight.
- Added an experimental new section in `config`, called `config:agentic`. This section can be enabled through the "apply_tools" option.
The section consists of a multi-step agentic system, where the the agents are in fixed order: ToolAgent (can search the internet, scrape, etc) -> Pythonagent (specialized in generating Python code) -> JudgeAgent. 
In the end, the final response is computed by a last agent. All agents are the currently loaded model with different roles.
- Added --log and --level flags to the CLI. This can be used to specify a log file and log level, like so:
```bash
owlsight --log log.txt --level DEBUG
```


If you encounter any issues, feel free to shoot me an email at v.ouwendijk@gmail.com