from transformers.pipelines import SUPPORTED_TASKS

TASK_TO_AUTO_MODEL = {
    k:v["pt"][0] for k,v in SUPPORTED_TASKS.items()
}

HUGGINGFACE_TASKS = [None] + [
    "text-generation",
    "text2text-generation",
    "translation",
    "summarization",
]
