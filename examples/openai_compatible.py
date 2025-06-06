import json
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from owlsight.processors.text_generation_processors import TextGenerationProcessorTransformers

app = FastAPI()
processor = TextGenerationProcessorTransformers("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


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


if __name__ == "__main__":
    # send request to the API
    # import requests
    # response = requests.post(
    #     "http://localhost:8000/v1/chat/completions",
    #     json={
    #         "messages": [
    #             {"role": "user", "content": "Write a short poem about a curious cat."}
    #         ],
    #         "stream": True,
    #         "max_new_tokens": 100
    #     }
    # )
    # print(response.text)
    print("Initializing TextGenerationProcessor with model 'mistralai/Mistral-7B-Instruct-v0.2'...")
    print("Note: Model loading by Hugging Face Transformers can take a few minutes")
    print("and consume significant RAM/disk space, especially on first download.")
    # The processor is initialized globally, which triggers model loading.
    # Adding a check to ensure it was initialized if we had more complex error handling for it.
    if 'processor' in globals() and processor is not None:
        print(f"Starting FastAPI server on http://localhost:8000")
        print("OpenAPI documentation (Swagger UI) available at http://localhost:8000/docs")
        print("You can send POST requests to http://localhost:8000/v1/chat/completions")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Error: Text generation processor was not initialized correctly.")
        print("Server cannot start.")
