from transformers.pipelines import SUPPORTED_TASKS
from transformers import QuestionAnsweringPipeline, DocumentQuestionAnsweringPipeline,  AutomaticSpeechRecognitionPipeline, VisualQuestionAnsweringPipeline, TranslationPipeline, SummarizationPipeline

TASK_TO_AUTO_MODEL = {k: v["pt"][0] for k, v in SUPPORTED_TASKS.items()}

HUGGINGFACE_TASKS = [None] + [
    "text-generation",
    "text2text-generation",
    "translation",
    "summarization",
]

HUGGINGFACE_MEDIA_TASKS = [
    "image-to-text",
    "automatic-speech-recognition",
    "visual-question-answering",
    "document-question-answering",
]

# MEDIA_PIPELINE_MAPPING = {
#     "image-to-text": "image",
#     "automatic-speech-recognition": "audio",
#     "visual-question-answering": "image",
#     "document-question-answering": "image",
# }

HUGGINGFACE_TASKS.extend(HUGGINGFACE_MEDIA_TASKS)
