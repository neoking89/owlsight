import sys
from pathlib import Path

sys.path.append("src")
from owlsight.configurations.schema import Schema
from owlsight.docs.helper_functions import get_init_docstrings, format_docstrings

# Get the path to the owlsight __init__.py file
init_path = Path(__file__).parent.parent / "__init__.py"

# Get API documentation
api_docs = get_init_docstrings(str(init_path))
formatted_api_docs = format_docstrings(api_docs)

README = f"""
# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models. It offers an interactive interface that allows you to execute Python code, shell commands, and use an AI assistant in one unified environment. This tool is ideal for those who want to integrate Python with generative AI capabilities.

## Why owlsight?

Picture this: you are someone who dabbles in Python occasionally. Or you frequently use generative AI to accelerate your workflow, especially for generating code. But often, this involves a tedious process—copying and pasting code between ChatGPT and your IDE, repeatedly switching contexts.

What if you could eliminate this friction?

Owlsight brings Python development and generative AI together, streamlining your workflow by integrating them into a single, unified platform. No more toggling between windows, no more manual code transfers. With Owlsight, you get the full power of Python and AI, all in one place—simplifying your process and boosting productivity. Owlsight has been designed to be a swiss-army knife for Python and AI with a core focus on open-source models, allowing you to execute code directly from model prompts and access this code directly from the Python interpreter.

## Key Features

### Interactive Environment
- **Command-Line Interface**: Choose from multiple commands such as Python, shell, and AI model queries
- **Python Integration**: Switch to a Python interpreter and use Python expressions in language model queries
- **Model Flexibility**: Support for models in **pytorch**, **ONNX**, and **GGUF** formats

### AI Integration
- **Huggingface Hub**: Search and download models directly from the CLI
- **Multimodal Support**: Work with models specialized for images and audio input (transformers-based models)
- **Retrieval Augmented Generation (RAG)**: Enhance prompts with Python library documentation

### Developer Tools
- **Customizable Configuration**: Easily modify model settings with intuitive save/load of config files
- **API Access**: Use Owlsight as a library in Python scripts, accessing the core CLI backend
- **Advanced Model Settings**: Fine-tune configuration parameters for optimal results

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

When working offline, you can use the offline flag. 
This will enable access to the tika-server.jar file locally, enabling you to use the DocumentReader class (which includes Apache Tika functionality) without an internet connection.
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

### Keyboard Shortcuts

When working with the editable option, the following keyboard shortcuts are available:

- **Ctrl+A**: Select all text in the current editable field
- **Ctrl+C**: Copy selected text
- **Ctrl+Y**: Paste selected text

### Getting Started

Now, lets start out by loading a model. Go to **config > huggingface**, choose a task like *text-generation* and press ENTER. 

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
How can I assist you? > How much is {{{{a}}}} * 5?
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
* **owl_press(sequence: List[str], exit_python_before_sequence: bool = True, time_before_sequence: float = 0.5, time_between_keys: float = 0.12)**
)**
  Press a sequence of keys in the terminal. This can be used to automate tasks or keypresses.
  - *sequence*: A list of keys to press. Available keys: 'L' (left), 'R' (right), 'U' (up), 'D' (down), 'ENTER' (ENTER), 'SLEEP:[float]' (sleep for time seconds), 'CTRL+A' (Select all), 'CTRL+C' (Copy), 'CTRL+Y' (Paste), DEL (Delete).
  - *exit_python_before_sequence*: If True, exit the Python interpreter after pressing the sequence.
  - *time_before_sequence*: Time to wait before pressing the first key.
  - *time_between_keys*: Time to wait between pressing each key.
* **owl_save_namespace(file_path: str)**
  Save all variables in the current namespace to a file, using the "dill" library.
  - *file_path*: The path to the file to save the namespace to.
* **owl_load_namespace(file_path: str)**
  Load all variables from a file into the current namespace, using the "dill" library.
* **owl_tools()**
  Display a list of available functions in the current namespace, which can be used for tool calling. All functions in the namespace are automatically converted to a fitting JSON-format in OPENAI's format.

## API Documentation

The following section details all the objects and functions available in the Owlsight API:

{formatted_api_docs}

## Configurations

Owlsight uses a configuration file in JSON-format to adjust various parameters. The configuration is divided into five main sections: `main`, `model`,  `generate`, `rag` and `huggingface`. Here's an overview of the application architecture:

{Schema.generate_diagram()}

Here's an example of what the default configuration looks like:

```json
{Schema.get_config_defaults(as_json=True)}
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
    model_id=r"path\to\Phi-3-mini-128k-instruct.Q5_K_S.gguf",
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
- Added **DocumentSearcher** to offer a general RAG solution for documents. At its core, uses a combination of TFIDF and Sentence Transformer.
- Added caching possibility to all RAG solutions in the API (*cache_dir* & *cache_dir_suffix*), where documents, embeddings etc. get pickled. This can save a big amount of time if amount of documents is large.

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
- Added possibility to pass complete directories as argument to mediatypes to a model in the CLI, like so: [[image:directory/containing/images]]
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
- ESC + V can be used inside the "How can I assist you?"-option after typing the following: "[[", "{{{{". This will autocomplete the following:
"[[" will autocomplete to: "image:", "audio:"
"{{{{" will autocomplete any available defined objects from the python-namespace.
- Added `owl_tools` function to the Python interpreter. This function can be used to convert all defined functions in the namespace to a dictionary, which can be used for tool/function-calling.
- Bracket-syntax "{{{{}}}}" for augmenting Python expressions can now also be used inside the `config` section of the CLI. For example, in the Python interpreter, we can store a long string inside a variable and pass it to `config:model:system_prompt` directly.
- Added new option `dynamic_system_prompt` to config:main section. This option can be used to dynamically generate a fitting system prompt first for a given user input, before passing it to the model.
The idea is that this might help the model to give a more focused response to the question.
- Add basic functionality, like select all, copy and paste. Use CTRL+A, CTRL+C and CTRL+Y respectively. This option applies to all editable fields and the Python Interpreter.

If you encounter any issues, feel free to shoot me an email at v.ouwendijk@gmail.com""".strip()


def write_readme(content: str, filename: str):
    """
    Write the README content to a file with proper encoding.

    Args:
        content (str): The content to write to the README
        filename (str): The output filename, defaults to README.md
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully wrote content to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    write_readme(README, "README.md")
