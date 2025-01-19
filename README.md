# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models. It offers an interactive interface that allows you to execute Python code, shell commands, and use an AI assistant in one unified environment. This tool is ideal for those who want to integrate Python with generative AI capabilities.

## Why owlsight?

Picture this: you are someone who dabbles in Python occasionally. Or you are a seasoned Pythonista. You frequently use generative AI to accelerate your workflow, especially for generating code. But often, this involves a tedious process—copying and pasting code between ChatGPT and your IDE, repeatedly switching contexts.

What if you could eliminate this friction?

Owlsight brings Python development and generative AI together, streamlining your workflow by integrating them into a single, unified platform. No more toggling between windows, no more manual code transfers. With Owlsight, you get the full power of Python and AI, all in one place—simplifying your process and boosting productivity.

Generate code directly from model prompts and access this code directly from the Python interpreter. Or augment model-prompts with Python expressions. With this functionality, open-source models do not only generate more accurate responses by executing Python code directly, but they can also solve way more complex problems.

## Features

- **Interactive CLI**: Choose from multiple commands such as Python, shell, and AI model queries.
- **Python Integration**: Switch to a Python interpreter and use python expressions in language model queries.
- **Model Flexibility**: Supports models in **pytorch**, **ONNX**, and **GGUF** formats.
- **Customizable Configuration**: Easily modify model and generation settings.
- **Retrieval Augmented Generation (RAG)**: Enrich prompts with documentation from Python libraries.
- **API Access**: Use Owlsight as a library in Python scripts.
- **Multimodal Support**: Use models that require additional input like images, audio, or video.

## Installation

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

To install all packages:

```
pip install owlsight[all]
```

It is recommended to use the `all` option, as this will install all dependencies and allow you to use all features of Owlsight.

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

Then, a distinction needs to be made in Owlsight between 3 different, but very simple option styles:

1. **Action**: This is just very simply an action which is being triggered by standing on an option in the menu and pressing ENTER.
   Examples from the main menu are:

   - *python*: Enter the python interpreter.
   - *clear history*: clear cache -and chat history.
   - *quit*: exit the Owlsight application.
2. **Toggle:** When standing on a toggle style option, press the LEFT and RIGHT arrow keys to toggle between different "multiple choice" options.
   Examples from the main menu are:

   - *config*: Toggle between the main, model, generate and rag config settings.
   - Inside the *config* settings, several other toggle options can be found. An easy example are the configurations where one can toggle between True and False.

     For more information about the config settings, read further down below the **Configurations** chapter.
3. **Editable:** This means the user can type in a text and press ENTER. This is useful for several situations in the mainmenu, like:

   - *how can I assist you?* : Given a model has been loaded by providing a valid *model_id*  in *config:model*,  type a question or instruction and press ENTER to get a response from the model.
   - *shell:* Interactive shell session. Type in a command and press ENTER.
   - *save*: Provide a valid path to save the current configurations as json. Then press ENTER. This is incredibly useful, as it allows later reuse of the current model with all its respective settings.
   - *load:* Provide a valid path to load configurations from an earlier saved json. Then press ENTER. If on windows, you can directly press ENTER without specifying a path to open up a file dialog window for convenience.

Now, lets start out by loading a model. Go to **config > huggingface** , choose a task like *text-generation* and press ENTER. 

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
How can I assist you? > How much is {a} * 5?
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

In Owlsight 2, models are supported that require additional input, like images or audio. In the backend, this is made possible with the **MultiModalProcessorTransformers** class. In the CLI, this can be done by setting the *model_id* to a multimodal model from the Huggingface modelhub. The model should be a Pytorch model. For convenience, it is recommended to select a model through the new Huggingface API in the configuration-settings (read below for more information).

The following tasks are supported:

- image-to-text
- automatic-speech-recognition
- visual-question-answering
- document-question-answering

These models require additional input, which can be passed in the prompt. The syntax for passing mediatypes done through special double-square brackets syntax, like so:

```
[[mediatype:path/to/file]]
```

The supported mediatypes are: *image*, *audio*.
For example, to pass an image to a document-question-answering model, you can use the following syntax:

```
What is the first sentence in this image? [[image:path/to/image.jpg]]
```

## Python interpreter

Next to the fact that objects generated by model-generated code can be accessed, the Python interpreter also has some useful default functions, starting with the "owl_" suffix. These serve as utilityfunctions.

These are:

* **owl_import(file_path: str)**
  Import a Python file and load its contents into the current namespace.
  - *file_path*: The path to the Python file to import.
* **owl_read(file_path: str)**
  Read the content of a text file.
  - *file_path*: The path to the text file to read.
* **owl_scrape(url_or_terms: str, trim_newlines: int = 2, filter_by: Optional[dict], request_kwargs: dict)**
  Scrape the text content of a webpage or search Bing and return the first result as a string.
  * `url_or_terms`: Webpage URL or search term.
  * `trim_newlines`: Max consecutive newlines (default 2).
  * `filter_by`: Dictionary specifying HTML tag and/or attributes to filter specific content.
  * `**request_kwargs`: Additional options for `requests.get`.
* **owl_show(docs: bool = False)**
  Display all imported objects (optional: include docstrings).
  - *docs*: If True, also display docstrings.
* **owl_write(file_path: str, content: str)**
  Write content to a text file.
  - *file_path*: The path to the text file to write.
  - *content*: The content to write to the file.
* **owl_history(to_string: bool = False)**
  Display command history (optional: return as string).
  - *to_string*: If True, returns the history as a formatted string, by default False
* **owl_models(cache_dir: str = None, show_task: bool = False)**
  Display all Hugging Face models currently loaded in the cache directory. Shows model names, sizes, and last modified dates.
  * `cache_dir`: Optional path to custom cache directory. If None, uses default Hugging Face cache.
  * `show_task`: If True, also displays the task associated with each model (may take longer to load).
* **owl_press(sequence: List[str], exit_python_from_interpreter: bool = True, time_before_sequence: float = 0.5, time_between_keys: float = 0.12)**
)**
  Press a sequence of keys in the terminal. This can be used to automate tasks or keypresses.
  - *sequence*: A list of keys to press. Available keys: 'L' (left), 'R' (right), 'U' (up), 'D' (down), 'ENTER' (ENTER), 'SLEEP:[float]' (sleep for time seconds).
  - *exit_python_from_interpreter*: If True, exit the Python interpreter after pressing the sequence.
  - *time_before_sequence*: Time to wait before pressing the first key.
  - *time_between_keys*: Time to wait between pressing each key.
* **owl_save_namespace(file_path: str)**
  Save all variables in the current namespace to a file, using the "dill" library.
  - *file_path*: The path to the file to save the namespace to.
* **owl_load_namespace(file_path: str)**
  Load all variables from a file into the current namespace, using the "dill" library.

## API Documentation

The following section details all the objects and functions available in the Owlsight API:


### Classes

#### TextGenerationProcessorOnnx

```python
class TextGenerationProcessorOnnx(model_id: str, onnx__verbose: bool = False, onnx__n_cpu_threads: int = 8, onnx__model_dir: Optional[str] = None, token: Optional[str] = None, apply_chat_history: bool = False, system_prompt: str = None, **kwargs)
```

Text generation processor using ONNX Runtime with HuggingFace Hub support.

**Methods:**

- `apply_chat_template(self, input_data: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer) -> str`
  - Apply chat template to the input text.
- `generate(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, stopwords: Optional[List[str]] = None, buffer_wordsize: int = 10, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text using the ONNX model.
- `generate_stream(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.0, generation_kwargs: Optional[Dict[str, Any]] = None)`
  - Stream generated text token by token.
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
class TextGenerationProcessorTransformers(model_id: str, transformers__device: Optional[str] = None, transformers__quantization_bits: Optional[int] = None, transformers__stream: bool = True, transformers__model_kwargs: Optional[dict] = None, bnb_kwargs: Optional[dict] = None, tokenizer_kwargs: Optional[dict] = None, task: Optional[str] = None, apply_chat_history: bool = False, system_prompt: str = '', **kwargs)
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
class TextGenerationProcessorGGUF(model_id: str, gguf__filename: str = '', gguf__verbose: bool = False, gguf__n_ctx: Optional[int] = None, gguf__n_gpu_layers: int = 0, gguf__n_batch: Optional[int] = None, gguf__n_cpu_threads: Optional[int] = None, apply_chat_history: bool = False, system_prompt: str = '', model_kwargs: Dict[str, Any] = None, **kwargs)
```

Text generation processor using GGUF models. Uses llama-cpp.Llama class under the hood.

**Methods:**

- `apply_chat_template(self, input_data: str) -> List[Dict[str, str]]`
  - Apply chat template to the input text.
- `generate(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.1, stopwords: Optional[List[str]] = None, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text based on input data.
- `generate_stream(self, input_data: str, max_new_tokens: int = 512, temperature: float = 0.1, generation_kwargs: Optional[Dict[str, Any]] = None)`
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self) -> Optional[int]`
  - Retrieve the maximum context length of the model.
- `update_history(self, input_data: str, generated_text: str) -> None`
  - Update the history with the input and generated text.

#### MultiModalProcessorTransformers

```python
class MultiModalProcessorTransformers(model_id: str, task: str, apply_chat_history: bool = False, system_prompt: str = '', **kwargs)
```

MultiModalProcessorTransformers class for multimodal text generation with Hugging Face models.
Supports image, audio inputs.

**Methods:**

- `apply_chat_template(self, input_data: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer) -> str`
  - Apply chat template to the input text.
- `generate(self, input_data: str, media_objects: Dict[str, owlsight.utils.custom_classes.MediaObject], stopwords: Optional[List[str]] = None, max_new_tokens: int = 512, temperature: float = 0.0, generation_kwargs: Optional[Dict[str, Any]] = None) -> str`
  - Generate text based on input data.
- `get_history(self) -> List[Dict[str, str]]`
  - Get complete chat history of inputs and outputs and system prompt.
- `get_max_context_length(self)`
  - Retrieve the maximum context length of the model.
- `preprocess_input(self, input_data: Union[str, bytes, pathlib.Path], question: Optional[str] = None) -> Any`
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
- `search(self, library: str, query: str, top_k: int = 5, cache_dir: Optional[str] = None, return_context: bool = True, tfidf_weight: float = 1.0, sentence_transformer_weight: float = 0.0, sentence_transformer_model: str = 'Alibaba-NLP/gte-base-en-v1.5', sentence_transformer_batch_size: int = 64) -> Union[pandas.core.frame.DataFrame, str]`
  - Search Python library documentation with caching for documents and search engines.

#### DocumentSearcher

```python
class DocumentSearcher(documents: Dict[str, str], sentence_transformer_model: str = 'Alibaba-NLP/gte-base-en-v1.5', sentence_transformer_batch_size: int = 64, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None)
```

A generic class for searching documents using an ensemble of TFIDF and Sentence Transformer methods.

**Methods:**

- `search(self, query: str, top_k: int = 20, sentence_transformer_weight: float = 0.7, tfidf_weight: float = 0.3) -> pandas.core.frame.DataFrame`
  - Search documents using the configured ensemble methods.

#### HashingVectorizerSearchEngine

```python
class HashingVectorizerSearchEngine(documents: Dict[str, str], cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, **hashing_kwargs)
```

Hashing Vectorizer based search implementation.

**Methods:**

- `create_index(self) -> None`
  - Create search index.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Perform search operation.

#### TFIDFSearchEngine

```python
class TFIDFSearchEngine(documents: Dict[str, str], cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, **tfidf_kwargs)
```

TF-IDF based search implementation.

**Methods:**

- `create_index(self) -> None`
  - Create search index.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Perform search operation.

#### SentenceTransformerSearchEngine

```python
class SentenceTransformerSearchEngine(documents: Dict[str, str], model_name: str = 'Alibaba-NLP/gte-base-en-v1.5', pooling_strategy: Literal['mean', 'max', None] = 'mean', device: Optional[str] = None, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None, batch_size: int = 64)
```

Sentence Transformer based search implementation.

**Methods:**

- `create_index(self) -> None`
  - Create search index with optimized batch processing.
- `get_full_cache_path(self) -> pathlib.Path`
  - Get full cache path.
- `get_suffix_filename(self) -> str`
  - Get the suffix filename.
- `load_data(self) -> Optional[Any]`
  - Load data from cache.
- `save_data(self, data: Any)`
  - Save data to cache.
- `search(self, query: str, top_k: int = 3) -> List[owlsight.rag.custom_classes.SearchResult]`
  - Perform search operation.
- `split_and_clean_text(text: str) -> List[str]`
  - Split a longer text into sentences and clean them.

#### OwlDefaultFunctions

```python
class OwlDefaultFunctions(globals_dict: owlsight.utils.custom_classes.SingletonDict)
```

Define default functions that can be used in the Python interpreter.
This provides the user with some utility functions to interact with the interpreter.
Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

This class is open for extension, as possibly more useful functions can be added in the future.

**Methods:**

- `owl_import(self, file_path: str)`
  - Import a Python file and load its contents into the current namespace.
- `owl_load_namespace(self, file_path: str)`
  - Load namespace using dill.
- `owl_models(self, cache_dir: Optional[str] = None, show_task: bool = False) -> str`
  - Returns a string with information about all Hugging Face models currently loaded in the cache directory.
- `owl_press(self, sequence: List[str], exit_python_from_interpreter: bool = True, time_before_sequence: float = 0.5, time_between_keys: float = 0.12) -> bool`
  - Simulate typing a sequence of keys and automaticly control the menu inside the Owlsight application.
- `owl_read(self, file_path: str) -> str`
  - Read the content of a text file.
- `owl_save_namespace(self, file_path: str)`
  - Save the current python namespace using dill.
- `owl_scrape(self, url_or_terms: str, trim_newlines: Optional[int] = 2, filter_by: Optional[Dict[str, str]] = None, **request_kwargs) -> str`
  - Scrape the text content of a webpage and return specific content based on the filter.
- `owl_show(self, docs: bool = True)`
  - Show all currently active imported objects in the namespace except builtins.
- `owl_write(self, file_path: str, content: str)`
  - Write content to a text file.

#### SystemPrompts

```python
class SystemPrompts()
```

System prompts for different expert roles

**Methods:**

- `as_dict(self) -> Dict[str, str]`

#### PromptWriter

```python
class PromptWriter(prompt: str)
```

Writes a system prompt to an Owlsight configuration JSON file.

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

Checks if a CUDA-capable GPU is available and if CUDA is installed.

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

#### search_bing

```python
def search_bing(term: str, exclude_from_url: Optional[List] = None, **request_kwargs) -> list
```

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
    - track_model_usage: Show metrics, which tracks GPU/CPU usage, amount of generated words and responsetime of model, Options: False, True, Type: OptionType.TOGGLE
    - extra_index_url: Additional URL for Python package installation. Useful for example when installing python packages (through pip) from private repositories, Type: OptionType.EDITABLE
    - sequence_on_loading: A list of key sequences to execute when loading the configuration. Uses owl_press functionality., Type: OptionType.EDITABLE
  - model settings:
    - back: Return to previous menu
    - model_id: Model identifier or path. The most important parameter in the configuration, as this will load the model to be used, Type: OptionType.EDITABLE
    - apply_chat_history: Whether to apply chathistory to the model prompt. All chathistory is saved as default, but when this is True, This history is added to the model prompt, Options: False, True, Type: OptionType.TOGGLE
    - system_prompt: System prompt defining model behavior, Type: OptionType.EDITABLE
    - transformers__device: Device for transformers model, Options: None, cpu, cuda, mps, Type: OptionType.TOGGLE
    - transformers__quantization_bits: Quantization bits for transformers model, Options: None, 4, 8, 16, Type: OptionType.TOGGLE
    - transformers__stream: Whether to stream input to transformers model, Options: False, True, Type: OptionType.TOGGLE
    - transformers__model_kwargs: Additional model parameters for transformers model, Type: OptionType.EDITABLE
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
    - stopwords: Stopwords that stop text generation. This can be useful for getting more control over when modelgeneration should stop, Type: OptionType.EDITABLE
    - max_new_tokens: Maximum amount of tokens to generate, Options: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, Type: OptionType.TOGGLE
    - temperature: Temperature for model generation, Options: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, Type: OptionType.TOGGLE
    - generation_kwargs: Additional generation parameters, like top_k, top_p, etc, Type: OptionType.EDITABLE
  - rag settings:
    - back: Return to previous menu
    - active: Whether RAG for python libraries is active. If True, the search-results will be implicitly added as context to the modelprompt and when pressing ENTER, search-results will be shown, Options: False, True, Type: OptionType.TOGGLE
    - target_library: Target python library for to use for RAG. If the library is not installed in the active environment, a warning will be showed with available options, Type: OptionType.EDITABLE
    - top_k: Number of most matching RAG results to return, based on `search` query, Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, Type: OptionType.TOGGLE
    - sentence_transformer_weight: Weight for embedding model, which is Alibaba-NLP/gte-base-en-v1.5. TFIDF-weight is 1 - `sentence_transformer_weight`, Options: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, Type: OptionType.TOGGLE
    - search: RAG search query. Press ENTER to show the `top_k` results. Only used when `active` is True, Type: OptionType.EDITABLE
  - huggingface settings:
    - back: Return to previous menu
    - search: Search for a model on the Hugging Face Hub by pressing ENTER. Keywords can be used optionally to finetune searchresults, e.g. 'llama 3b gguf', Type: OptionType.EDITABLE
    - top_k: Top number of Hugging Face results to return. The results will be sorted by highest score first, Options: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, Type: OptionType.TOGGLE
    - select_model: Select and load a model from the Hugging Face Hub by toggling through the options found by `search`, Type: OptionType.TOGGLE
    - task: Filter Hugging Face models by task. When using `search`, the results will be filtered directly by chosen task, Options: None, text-generation, text2text-generation, translation, summarization, image-to-text, automatic-speech-recognition, visual-question-answering, document-question-answering, Type: OptionType.TOGGLE
- save: Save current configuration as JSON-file
- load: Load a configuration from a JSON-file
- clear history: Clear cache and chat history
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
        "sequence_on_loading": []
    },
    "model": {
        "model_id": "",
        "apply_chat_history": false,
        "system_prompt": "",
        "transformers__device": null,
        "transformers__quantization_bits": null,
        "transformers__stream": true,
        "transformers__model_kwargs": {},
        "gguf__filename": "",
        "gguf__verbose": false,
        "gguf__n_ctx": 512,
        "gguf__n_gpu_layers": 0,
        "gguf__n_batch": 8,
        "gguf__n_cpu_threads": 8,
        "onnx__model_dir": "",
        "onnx__verbose": false,
        "onnx__n_cpu_threads": 8
    },
    "generate": {
        "stopwords": [],
        "max_new_tokens": 512,
        "temperature": 0.0,
        "generation_kwargs": {}
    },
    "rag": {
        "active": false,
        "target_library": "",
        "top_k": 10,
        "sentence_transformer_weight": 0.0,
        "search": ""
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

## API

Owlsight can also be used as a library in Python scripts. The main classes are the `TextGenerationProcessor` family, which can be imported from the `owlsight` package. Here's an example of how to use it:

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

## RELEASE NOTES

**1.0.2**

- Enhanced cross-platform compatibility.
- Introduced the `generate_stream` method to all `TextGenerationProcessor` classes.
- Various minor bug fixes.

**1.1.0**

- Added Retrieval Augmented Generation (RAG) for enriching prompts with documentation from python libraries. This option is also added to the configuration.
- History with autocompletion is now also available when writing prompts. Prompts can be autocompleted with TAB.

**1.2.1**

- Access backend functionality through the API using "from owlsight import ..."
- Added default functions to the Python interpreter, starting with the "owl_" suffix.
- More configurations available when using GGUF models from the command line.

**1.3.0**

- Add `owl_history` function to python interpreter for directly accessing model chat history.
- Improved validation when  loading a configuration file.
- Added validation for retrying a codeblock from an error. This configuration is called `prompt_retry_on_error`

**1.4.1**

- improve RAG capabilities in the API, added **SentenceTransformerSearchEngine**, **TFIDFSearchEngine** and **HashingVectorizerSearchEngine** as classes.
- Added **DocumentSearcher** to offer a general RAG solution for documents.
- Added caching possibility to all RAG solutions in the API (*cache_dir* & *cache_dir_suffix*), where documents, embeddings etc. get pickled. This can save a big amount of time if amount of documents is large.

**2.0.1beta**

*BREAKING CHANGES*

- Added Huggingface API in the configuration-settings of the CLI. This allows the user to search and load models directly from the Huggingface modelhub and can be found through `config:huggingface`.
- added `transformers__use_fp16` and `transformers__stream` to `config:model` for using fp16 and streaming the model output in the transformers-based models.
- Added **MultiModalProcessorTransformers** for non text-input based models. This class can be used for models which require additional input like images, audio or video and works with models from the Huggingface Hub based on the Pytorch framework.
- Introduced new double-square brackets syntax for passing mediatypes in the prompt.
- Improved logging with clearer color coding and more detailed information.
- System Prompt is now an empty string as default.
- Several small bugfixes and improvements.

**2.0.2 (stable)**

- Upgraded UI with new color scheme and improved readability. Description of the current choice is now displayed above the menu.
- Removed `onnx__tokenizer` from `TextGenerationProcessorOnnx` constructor, so that only *model_id* is needed as constructor argument.
- Added `get_max_context_length` method to all `TextGenerationProcessor` classes, which returns the maximum context length of the loaded model.
- Moved `transformers__use_fp16` in config:model to `transformers__quantization_bits` as value 16, as it is more clear.
- Added `track_model_usage` to config:main, which can be used to track usage of the model, like the amount of words generated, total time spent etc.
- Added possibility to pass complete directories as argument to mediatypes, like so: [[image:directory/containing/images]]
- Add owl_models() function to python interpreter for displaying all Huggingface models in the cache directory.

**2.?**

- Add `owl_save_namespace` `owl_load_namespace` functions to save and load all variables inside the Python interpreter. This 
is useful if you want to save any code created by a model. Or load a namespace from a previous session.
- `ProcessorMemoryContext` can be used as a context_manager to clean up resources from `TextGenerationProcessor`, like the model, from memory after usage.
- Improved `config:rag` functionality with the new `sentence-transformer-weight` option. This allows to weigh the sentence-transformer part in the RAG model next to the already present TFIDF, improving semantic search capabilities.
- Added `main:sequence_on_loading` to the configuration json. This allows execution of a sequence of keys on loading a config through the `load` option in the Owlsight main-menu.
TIP: above option can be used to load a sequence of different models as "agents", where every config can be threaded as a different agent with their own role. In theory, every action in Owlsight can be automated through this option.

If you encounter any issues, feel free to shoot me an email at v.ouwendijk@gmail.com