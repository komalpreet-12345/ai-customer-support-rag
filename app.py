from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ask_bot

app = FastAPI(title="Customer Support Bot")

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    response = ask_bot(query.message)
    return {"response": response}