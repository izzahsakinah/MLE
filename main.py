from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from fastapi import Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Read the key from env
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Setup Chroma client and model
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("countries")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample in-memory data for quick testing
data = [
    "Germany, capital is Berlin, population is 83 million, area is 357,022 km².",
    "Japan, capital is Tokyo, population is 126 million, area is 377,975 km²."
]

@app.post("/refresh-data")
def refresh_data():
    try:
        # Clear existing data in the collection
        collection.delete(where={"document": {"$ne": "___DELETE_ME___"}}) # delete all documents
        for i, text in enumerate(data):
            # Encode the text and add to the collection
            # Generate a unique ID for each document to avoid conflicts
            embedding = model.encode(text).tolist()
            collection.add(documents=[text], embeddings=[embedding], ids=[f"country_{i}"])
        return {"status": "Chroma DB refreshed."}
    except Exception as e:
        # Log the error and return a message
        # This will help in debugging if something goes wrong
        print("Error in /refresh-data:", e)
        return {"error": str(e)}


@app.get("/query")
def query_data(question: str):
    question_embedding = model.encode(question).tolist()
    # Query the collection with the question embedding
    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    return results

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Encode the question and query the collection
        question_embedding = model.encode(req.question).tolist()
        result = collection.query(query_embeddings=[question_embedding], n_results=3)
        context = "\n".join(result["documents"][0])
        
        # If no context found, return a default message
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer only using the context below. Say 'I don't know' if the answer is missing."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
            ]
        )

        # Return the response content
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        # Log the error and return a message
        # This will help in debugging if something goes wrong
        print("ERROR in /chat:", e)
        return {"error": str(e)}
