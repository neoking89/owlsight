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
json_schema = Schema.get_config_defaults(as_json=True)
schema_diagram = Schema.generate_diagram()

README_INTRO = """
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

### Installation Options and Dependencies

A basic installation includes only the core dependencies needed for the transformers library and basic functionality. For access to specific features, you will need to install optional dependency groups:

#### Optional Feature Modules

To add GGUF model support (using llama-cpp-python):
```
pip install owlsight[gguf]
```

To add ONNX model support (optimized model inference):
```
pip install owlsight[onnx]
```

To add multimodal functionality (image processing, OCR):
```
pip install owlsight[multimodal]
```

To add web search and scraping capabilities:
```
pip install owlsight[search]
```

To add voice control functionality:
```
pip install owlsight[voice]
```

For operating in an offline environment with tika-server.jar file, enabling you to use the `DocumentReader` class (which includes Apache Tika functionality):
```
pip install owlsight[offline]
```

#### Comprehensive Installation
To install all packages and features:
```
pip install owlsight[all]
```

### Available Features Based on Installation

| Feature | Basic Install | Required Extra |
|---------|---------------|----------------|
| Transformers models | ✓ | - |
| GGUF models | ✗ | [gguf] |
| ONNX models | ✗ | [onnx] |
| Image processing | ✗ | [multimodal] |
| Web search/scraping | ✗ | [search] |
| Voice control | ✗ | [voice] |
| Offline document reading (using Apache Tika) | ✗ | [offline] |
| Development | ✗ | [dev] |

### Security and Performance Notes

- Using multiple flags is a conscious design choice to give users more control over the behavior of the application and prevent "dependency hell".
- The application is designed to gracefully handle missing dependencies - you will receive helpful warning/error messages if you attempt to use a feature without the required dependencies.
- Some libraries like llama-cpp-python and pytorch may require specific configurations depending on your hardware.
- If you want most useful features out of the box, it is recommended to pip install Owlsight with the [all] option. This will install owlsight with the following flags: gguf, onnx, multimodal, search
- Recommended python versions: 3.10, 3.11, 3.12. Lower or higher python versions may not support all features, especially due to package incompatibilities.

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

### Voice Control

Owlsight supports voice control functionality when installed with `pip install owlsight[voice]`. This allows you to control the application using voice commands.

To enable voice control, use the `--voice` flag when starting Owlsight:
```bash
owlsight --voice
```

You can customize the voice control behavior using JSON-based configuration:

```bash
# Custom key mappings (spoken words to keyboard actions)
owlsight --voice --word-to-key '{
    "backward": "left",
    "forward": "right",
    "save": ["ctrl", "s"],
    "select all": ["ctrl", "a"]
}'

# Custom word substitutions
owlsight --voice --word-to-word '{
    "print": "print()",
    "function": "def my_function():",
    "exit": "exit()"
}'

# Advanced voice control settings
owlsight --voice --voicecontrol-kwargs '{
    "cmd_cooldown": 0.5,
    "debug": true,
    "language": "en",
    "model": "base.en",
    "key_press_interval": 0.1,
    "typing_interval": 0.05
}'
```

These options can be combined to create a fully customized voice control experience, which you can also utilize outside of the application.

### Example Workflow

You can combine Python variables defined in the Python Interpreter together with language models in Owlsight through special double curly-brackets syntax.
For example:

```
python > a = 42
How can I assist you? > How much is {{a}} * 5?
```

```
answer -> 210
```

Additionally, you can also ask a model to write pythoncode and access that in the python interpreter.

From a model response, all generated python code will be extracted and can be edited or executed afterwards. This choice is always optional. After execution, the defined objects will be saved in the global namespace of the python interpreter for the remainder of the current active session. This is a powerful feature, which allows build-as-you-go for a wide range of complex tasks.

Example:

```
How can I assist you? > Can you write a function which reads an Excel file?
```

-> *model writes a function called read_excel*

```
python > excel_data = read_excel("path/to/excel")
```

### MultiModal Support

In Owlsight 2, special multimodal support is available for certain models that require additional input, like images or audio. In the backend, this is made possible with the **MultiModalProcessorTransformers** class. 
In the CLI, this can be done by setting the *config.model.model_id* to a multimodal model from the Huggingface modelhub. 
Keep in mind that this model should be a Pytorch model (so not GGUF or ONNX).
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
- `owl_search_and_scrape`: Search and scrape the web using DuckDuckGo's API. Uses both the `owl_search` and `owl_scrape` functions combined.
- `owl_create_document_searcher`: Create a DocumentSearcher instance with a given set of documents and a text splitter. This class is great for usage in a RAG scenario.
"""
CONFIGURATION = f"""
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
"""

API_EXAMPLES = """
## API Examples

Owlsight can also be used as a library in Python scripts. The main classes are the `TextGenerationProcessor` family, which can be imported from the `owlsight` package. 

Here is a simple example of how to use it:
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

Alternatively, there is a lot more to explore in the `owlsight` package.
Here is an example on how to use the `DocumentSearcher` class for simple document retrieval:
```python
from owlsight import DocumentSearcher, SentenceTextSplitter, SemanticTextSplitter

docs = {
    "doc1": "Quantum mechanics describes nature at atomic scales, introducing wave-particle duality and entanglement.",
    "doc2": "General relativity redefines gravity as spacetime curvature, predicting black holes and gravitational waves.",
    "doc3": "Quantum gravity aims to unify quantum mechanics and relativity, with theories like string theory and LQG.",
    "doc4": "String theory is a framework for understanding the universe, with models like the Minkowski space-time and the Einstein-Hilbert action.",
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

Or a more advanced example of similarity search, where some websites are scraped and being splitted in chunks based on their semantic similarity.
```python
from owlsight import OwlDefaultFunctions, SemanticTextSplitter, DocumentSearcher

if __name__ == "__main__":
    owl_funcs = OwlDefaultFunctions({})

    # List of AI/ML related URLs to scrape
    urls = [
        "https://plato.stanford.edu/entries/artificial-intelligence/",  # Stanford's AI Philosophy
        "https://www.nature.com/articles/s42256-019-0088-2",  # Nature's Deep Learning overview
    ]

    scraped_text = owl_funcs.owl_scrape(urls)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    doc_splitter = SemanticTextSplitter(percentile=0.99, target_chunk_length=400, model_name=model_name)
    doc_searcher = DocumentSearcher(scraped_text, sentence_transformer_model=model_name, text_splitter=doc_splitter)
    df = doc_searcher.search("reinforcement learning")
    seperator = "-" * 100
    for idx, row in enumerate(df.iterrows(), start=1):
        print(seperator)
        score = row[1]["aggregated_score"]
        print(f"Rank: {idx} (Score: {score:.2f})")
        print(f"Document name: {row[1]['document_name']}")
        print(row[1]["document"])

```
"""

API_DOCUMENTATION = f"""
## API Documentation

The following section details all the objects and functions available in the Owlsight API:

{formatted_api_docs}
"""

RELEASE_NOTES = """
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

**2.4.0.beta**

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
- Added `additional_information` option to the `config:agentic` section. This option can be used to add additional information to every agent call, for example: "Do NOT use owl_scrape and owl_search, because there is no internet connection."
- Added voice control support with customizable mappings through `owlsight[voice]` package
This can be used for (close to realtime) transcription of user input to the screen, using faster-whisper.
Voice control features include:
  * Customizable word-to-key mappings for keyboard control
  * Word-to-word substitutions for text input
  * Configurable settings like command cooldown and typing intervals
  * Support for multiple languages and speech recognition models
- Added JSON-based configuration for all voice control settings
- Added `owl_search_and_scrape` function to the Python interpreter. This function can be used to search and scrape the web using DuckDuckGo's API.
- Added `owl_create_document_searcher` function to the Python interpreter. This utilityfunction can be used to create a `DocumentSearcher` instance with a given set of documents and a text splitter.

**2.4.0(stable)**
- Added `get_mteb_leaderboard_data` function to the backend API. This function can be used to fetch the MTEB leaderboard data.
- Added support for `uv` as an alternative package manager. Also improved current support for `pip` environments.
- Several minor bugfixes and improvements.



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
    seperator = "\n\n"
    README = "".join(
        [
            README_INTRO.strip(),
            seperator + CONFIGURATION.strip(),
            seperator + API_EXAMPLES.strip(),
            seperator + API_DOCUMENTATION.strip(),
            seperator + RELEASE_NOTES.strip(),
        ]
    )
    write_readme(README, "README.md")
