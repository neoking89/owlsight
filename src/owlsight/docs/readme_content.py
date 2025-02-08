"""Module containing the static content for the README file."""

STATIC_CONTENT = """
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
   - *clear history*: clear cache and chat history.
   - *quit*: exit the Owlsight application.

2. **Configuration**: This is a configuration option which can be edited by pressing ENTER on it. This will open a submenu with more options.
   Examples from the main menu are:

   - *config: main*: This opens a submenu with options for configuring the main application settings.
   - *config: model*: This opens a submenu with options for configuring the model settings.
   - *config: generate*: This opens a submenu with options for configuring the generation settings.
   - *config: rag*: This opens a submenu with options for configuring the RAG settings.
   - *config: huggingface*: This opens a submenu with options for configuring the Hugging Face settings.

3. **Editable:** This means the user can type in a text and press ENTER. This is useful for several situations in the mainmenu, like:

   - *how can I assist you?* : Given a model has been loaded by providing a valid *model_id*  in *config:model*,  type a question or instruction and press ENTER to get a response from the model.
   - *shell:* Interactive shell session. Type in a command and press ENTER.
   - *save*: Provide a valid path to save the current configurations as json. Then press ENTER. This is incredibly useful, as it allows later reuse of the current model with all its respective settings.
"""
