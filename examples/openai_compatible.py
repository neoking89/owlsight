from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from owlsight.processors.text_generation_processors import TextGenerationProcessorTransformers

app = FastAPI()
processor = TextGenerationProcessorTransformers("mistralai/Mistral-7B-Instruct-v0.2")

@app.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    if body.get("stream"):
        generator = processor.generate_openai_comp(
            body["messages"],
            **{k: v for k, v in body.items() if k != "messages"},
        )
        return StreamingResponse(
            (f"data: {json.dumps(chunk)}\n\n" for chunk in generator),
            media_type="text/event-stream",
        )
    else:
        response = processor.generate_openai_comp(
            body["messages"],
            **{k: v for k, v in body.items() if k != "messages"},
        )
        return JSONResponse(response)
