import gradio as gr
import re
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB client and model
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("countries")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample in-memory data for quick suggestions
examples = [
    "Which countries have population over 100 million?",
    "What is the capital of Afghanistan?",
    "List countries with area smaller than 10000 km²",
]

def extract_keyword(text):
    # Extract last word in query, e.g. "tell me about andorra" → "andorra"
    match = re.search(r"(?:about|on|of|for)\s+([\w\s]+)", text.lower())
    if match:
        return match.group(1).strip()
    return text.lower().strip()

# Function to handle user queries
def ask_bot(message, history):
    # If the message is empty, return a default response
    keyword = extract_keyword(message)
    #keyword = message.strip().lower()
    # Try both name and capital using partial matching
    country_match = collection.get(where={"name": keyword})
    capital_match = collection.get(where={"capital": keyword})


    # If we have an exact match for country name or capital
    if country_match["documents"]:
        context = country_match["documents"][0]
    # If we have an exact match for capital    
    elif capital_match["documents"]:
        context = capital_match["documents"][0]
    else:
        # If no exact full-text matching, use embedding to find relevant context
        message_embedding = model.encode(message).tolist()
        result = collection.query(query_embeddings=[message_embedding], n_results=252)

        #print("Retrieved documents:")
        #for doc in result["documents"][0]:
            #print(doc)

        # Safeguard if no documents returned
        if not result["documents"][0]:
            context = "No relevant data found."
        else:
            context = "\n".join(result["documents"][0])

    # If no context found, return a default message
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer using only the context. If the user asks about a capital, identify which country it belongs to"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}
        ]
    )

    # Return the response content
    return response.choices[0].message.content

# Set up Gradio interface
gr.ChatInterface( 
    fn=ask_bot,
    examples=examples,
    title="Country Data Chatbot",
).launch()
# This code sets up a Gradio interface for a chatbot that uses ChromaDB and OpenAI to answer questions based on country data.
# It initializes the necessary clients, defines a function to handle user queries, and launches the Gradio app.
# The chatbot retrieves relevant context from the ChromaDB collection and uses OpenAI's GPT-3.5-turbo model to generate responses.
# The Gradio interface allows users to interact with the chatbot in a user-friendly manner.
# The chatbot retrieves relevant context from the ChromaDB collection and uses OpenAI's GPT-3.5-turbo model to generate responses.