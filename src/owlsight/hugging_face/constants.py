from transformers.pipelines import SUPPORTED_TASKS

EXCLUDED_TASKS = [
    "feature-extraction",
    "table-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-audio-classification",
    "conversational",
    "zero-shot-object-detection",
]

HUGGINGFACE_TASKS = [None] + [task for task in SUPPORTED_TASKS if task not in EXCLUDED_TASKS]
