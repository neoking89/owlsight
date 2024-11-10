"Get list of supported tasks from Huggingface API."

from transformers.pipelines import SUPPORTED_TASKS
from .custom_classes import (
    TransformersArgumentInferer,
)

MODELHUB_PREFIX = "https://huggingface.co/"

EXCLUDED_TASKS = [
    "feature-extraction",
    "table-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-audio-classification",
    "conversational",
    "zero-shot-object-detection",
]

TASK_DICT = {task: SUPPORTED_TASKS[task] for task in SUPPORTED_TASKS if task not in EXCLUDED_TASKS}

for task in TASK_DICT.keys():
    pipeline_class = SUPPORTED_TASKS[task]["impl"]
    argument_inferer = TransformersArgumentInferer()
    TASK_DICT[task]["arguments"] = argument_inferer(pipeline_class.__call__)

TASK_DICT = dict(sorted(TASK_DICT.items(), key=lambda item: item[1]["type"][0::]))

INIT_KWARGS_PREFIX = (
    "Arguments for the transformers.pipeline() function.\nWill build a transformers.Pipeline object.\n\n"
)
CALL_KWARGS_PREFIX = "Arguments for the Pipeline.__call__ method.\n\n"
