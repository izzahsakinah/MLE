Country Data Chatbot

1. This project demonstrated the implementation of 

1. web scraping
2. data storage in ChromaDB
3. sentence embeddings using Hugging Face model
3. chatbot RAG using gradio and OpenAI with strict query scope

2. Setup requirements:

Python environment
- Python 3.9+
- create virtual environment then install dependencies:
- pip install :
chromadb
requests
beautifulsoup
sentence-transformers
openai
python-dotenv
gradio

3. ChromaDB to store country data as vector embeddings which:
- each country information such as name, population, area are stored as a single string
- sentence transformer converts each string into a vector embeddings
- embedding are stored in Chroma to ./chroma folder

4. Chatbot integration with RAG pipeline
- When users prompt a question such as "What is the capital of Afghanistan" 
- the exact match in metadata which is "name", "capital", "population", "area" detect as the context
- or (Else:), emdedding searched the query and retreived 252 documents to
- pass these documents to OpenAI via the query prompt "Answer using only the context. If the user asks about a capital, identify which country it belongs to"

5. API endpoints and chatbot usage guide
Backend (FASTAPI) for testing and debugging of API, Gradio, Chroma:
- /refresh-data to scrape and store country data to Chroma
- /query to get the query found in the documents (raw results)
= /chat retrieve the results from the documents and to display the output with OpenAI model (human-friendly results)

Frontend (Gradio) used (chosen due to suitability and easiness for designing a chatbot):
- gradio_app.py run on localhost
- sample questions are included: 
 "Which countries have population over 100 million?"
 "What is the capital of Afghanistan?"
 "List countries with area smaller than 10000 km²"

6. OpenAI API key
- add a .env file with OPENAI_API_KEY=sk...

7. Folder structure
project/
- web-scrapping.py (scrapping + chroma)
- main.py (testing and debugging using FastAPI before actual implementation)
- gradio_app.py 
- .env
- README.md
- chroma/
- venv/

8. Additional info
- Manual data storing for Andorra and UAE for system completeness

# MLE
