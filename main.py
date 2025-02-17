from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import uvicorn

# Create FastAPI app
app = FastAPI(title="T5 Translation API")


# Pydantic models for request/response
class TranslationRequest(BaseModel):
    sentences: list[str]


class TranslationResponse(BaseModel):
    translations: list[str]


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}


@app.post("/predict", response_model=TranslationResponse)
async def predict(request: TranslationRequest):
    """
    Translate English sentences to German using T5 model
    """
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    # Handle empty input case
    if not request.sentences:
        return TranslationResponse(translations=[])

    # Add task prefix to each sentence
    task_prefix = "translate English to German: "
    prefixed_sentences = [task_prefix + sentence for sentence in request.sentences]

    # Tokenize inputs
    inputs = tokenizer(prefixed_sentences, return_tensors="pt", padding=True)

    # Generate translations
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
    )

    # Decode translations
    translations = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    return TranslationResponse(translations=translations)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
