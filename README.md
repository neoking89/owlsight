# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models. It offers an interactive interface that allows you to execute Python code, shell commands, and natural language tasks in one unified environment. This tool is ideal for those who want to integrate Python with language model capabilities.

## Features

- **Interactive CLI**: Choose from multiple commands such as Python, shell, and AI model queries.
- **Python Integration**: Switch to a Python interpreter and use python expressions in language model queries.
- **Model Flexibility**: Supports models in **pytorch**, **ONNX**, and **GGUF** formats.
- **Customizable Configuration**: Easily modify model and generation settings.

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

## Usage

After installation, launch Owlsight in the terminal by running the following command:

```
owlsight
```

This will present you with the mainmenu:

```
Make a choice:
> how can I assist you?
shell
python
config: main
save
load
clear history
quit
```

Go to **config >** **model** and set a model_id to load a model locally or from *[https://huggingface.co/]()*

### Available Commands

* **How can I assist you**: Ask a question or give an instruction.
* **shell** : Execute shell commands.
* **python** : Enter a Python interpreter.
* **config: main** : Modify the *main*, *model* or *generate* configuration settings.
* **save/load** : Save or load a configurationfile.
* **clear history** : Clear the chathistory and the python interpreter history.
* **quit** : Exit the application.

### Example Workflow

You can combine Python variables with language models in Owlsight. For example:

```
python > a = 42
How can I assist you? > How much is {{a}} * 5?
```

```
answer -> 210
```

Additionally, one can also ask a model to write pythoncode and access that in the python interpreter. All defined objects will be saved in the global namespace of the python interpreter for the remainder of the current active session. This is a powerful feature, which allows build-as-you-go for a wide range of tasks.

Example:

```
How can I assist you? > Can you write a function which reads an excelfile?
```

-> *model writes a function called read_excel*

```
python > excel_data = read_excel("path/to/excel")
```

## Configurations

Owlsight uses a configuration file in JSON-format to adjust various parameters. Here is an example of what the configuration might look like:

```
{
    "main": {
        "max_retries_on_error": 5,
        "prompt_code_execution": true,
        "extra_index_url": ""
    },
    "model": {
        "model_id": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
        "save_history": true,
        "system_prompt": "# ROLE:\nYou are an advanced problem-solving AI with expert-level knowledge in various programming languages, particularly Python.\n\n# TASK:\n- Prioritize Python solutions when appropriate.\n- Present code in markdown format.\n- Clearly state when non-Python solutions are necessary.\n- Break down complex problems into manageable steps and think through the solution step-by-step.\n- Adhere to best coding practices, including error handling and consideration of edge cases.\n- Acknowledge any limitations in your solutions.\n- Always aim to provide the best solution to the user's problem, whether it involves Python or not.",
        "transformers__device": null,
        "transformers__quantization_bits": null,
        "gguf__filename": "Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf",
        "gguf__verbose": true,
        "gguf__n_ctx": 16384,
        "onnx__tokenizer": "",
        "onnx__verbose": true,
        "onnx__num_threads": 1
    },
    "generate": {
        "stopwords": [],
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "generation_kwargs": {}
    }
}
```

Configurationfiles can be saved and loaded through the mainmenu.

### Changing configurations

To update a configuration, simply modify the desired value and press **ENTER** to confirm the change. Please note that only one configuration setting can be updated at a time, and the change will only take effect once **ENTER** has been pressed.

## Temporary environment

During an Owlsight session, a temporary environment is created within the "site-packages" directory of the active (virtual) environment. Any packages installed during the session are removed when the session ends, ensuring your environment remains clean. If you want to persist installed packages, simply install them outside of Owlsight.

## Error Handling and Auto-Fix

Owlsight automatically tries to fix and retry any code that encounters a **ModuleNotFoundError** by installing the required package and re-executing the code. It can also attempt to fix errors in its own generated code. This feature can be controlled by the *max_retries_on_error* parameter in the configuration file.

## RELEASE NOTES

**1.0.2**

- Enhanced cross-platform compatibility.

* Introduced the `generate_stream` method to all `TextGenerationProcessor` classes.
* Various minor bug fixes.
* Enabled modular imports of individual components from the **owlsight** library, allowing direct usage of specific functionalities in Python scripts and applications.
