# Owlsight

**Owlsight** is a command-line tool that combines Python programming with open-source language models. It offers an interactive interface that allows you to execute Python code, shell commands, and natural language tasks in one unified environment. This tool is ideal for developers who want to seamlessly integrate Python with language model capabilities.

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


## Configuration

Owlsight uses a configuration file to adjust various parameters. Here is an example of what the configuration might look like:

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
