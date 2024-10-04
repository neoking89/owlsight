# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models. It offers an interactive interface that allows you to execute Python code, shell commands, and natural language tasks in one unified environment. This tool is ideal for those who want to seamlessly integrate Python with language model capabilities.

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

## Usage

After installation, launch Owlsight by running the following command:

```
pip install owlsight
```

This will present you with a menu like this:

```
Make a choice:
>how can I assist you?
shell
python
config: main
save
load
clear history
quit
```

### Available Commands

* **How can I assist you**: Ask a question or give an instruction.
* **shell** : Execute shell commands.
* **python** : Enter a Python interpreter.
* **config: main** : Modify the main configuration settings.
* **save/load** : Save or load a configurationfile.
* **clear history** : Clear the session history.
* **quit** : Exit the application.

### Example Workflow

You can combine Python variables with natural language processing models in Owlsight. For example:

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

## Configuration

Owlsight uses a configuration file in JSON-format to adjust various parameters. Here is an example of what the configuration might look like:

```
{
    "main": {
        "max_retries_on_error": 3,
        "prompt_code_execution": true,
        "extra_index_url": ""
    },
    "model": {
        "model_id": "path/to/microsoft-Phi3",
        "save_history": false,
        "system_prompt": "# ROLE:\nYou are an advanced problem-solving AI...",
        "transformers__device": null,
        "transformers__quantization_bits": null,
        "transformers__gguf_file": "",
        "onnx__tokenizer": "microsoft/Phi-3-mini-128k-instruct",
        "onnx__verbose": false,
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

## Temporary environment

When activated, Owlsight will create a temporary file during the remainder of the active session in the "Lib/site-packages" directory of the current active (virtual) environment. This is meant as a temporary container for installed packages during the active session. The idea behind this, is that all installed packages will be removed when the session ends, not clogging up the available memory. If one wants to persist installed packages, they can be simple be installed inside the active virtual environment outside of owlsight. 



## Fixing own code

When encountering a ModuleNotFoundError after executing a piece of code, Owlsight will automaticly try to install the package and execute the code again. Also, Owlsight provides an option to let the model fix and  retry its own generated code if faulty. This functionality can be controlled through the "max_retries_on_error" parameter in the config file.
