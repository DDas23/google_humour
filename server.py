import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load API Key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        properties={
            "Match_with_setup": content.Schema(type=content.Type.STRING),
            "Pun": content.Schema(type=content.Type.INTEGER),
            "Dark_Humor": content.Schema(type=content.Type.INTEGER),
            "Sarcasm": content.Schema(type=content.Type.INTEGER),
            "Wholesome": content.Schema(type=content.Type.INTEGER),
            "Overall_Humor_Percentage": content.Schema(type=content.Type.INTEGER),
            "Sentiment": content.Schema(type=content.Type.STRING),
        },
    ),
    "response_mime_type": "application/json",
}

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Request body schema
class JokeInput(BaseModel):
    setup: str
    punchline: str

@app.post("/analyze_humor")
async def analyze_humor(joke: JokeInput):
    try:
        # Format input for AI model
        prompt = f"input: \"Setup\": \"{joke.setup}\"\n\"Punchline\": \"{joke.punchline}\"\noutput: "
        
        # Generate response
        response = model.generate_content([prompt])
        return response.text  # Returns structured JSON

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Humor Analysis API is running!"}
