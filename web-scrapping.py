import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

def scrape_countries():
    url = "https://www.scrapethissite.com/pages/simple/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    
    # Find all country elements and extract relevant data
    # Assuming the structure of the HTML is known and contains country data
    countries = []
    country_elements = soup.find_all('div', class_='country')

    
    for element in country_elements:
        name = element.find('h3', class_='country-name').text.strip()
        capital = element.find('span', class_='country-capital').text.strip()
        population = element.find('span', class_='country-population').text.strip()
        area = element.find('span', class_='country-area').text.strip()
        
        # Create a single text chunk
        text_chunk = f"{name}, capital is {capital}, population is {population}, area is {area} km²"
        countries.append({
            'text': text_chunk,
            'metadata': {
                'name': name.lower(),
                'capital': capital.lower(),
                'population': int(population),
                'area': float(area)
            }
        })

    # Return the list of countries with their text and metadata
    # Adding manual entries for Andorra and United Arab Emirates
    manual = [
    {
        "text": "Andorra la Vella is the capital of Andorra, a country located in the Europe. Andorra, capital is Andorra la Vella has a population of 85000 and an area of 468 km²",
        "metadata": {
            "name": "andorra", 
            "capital": "andorra la vella", 
            "population": 85000, 
            "area": 468.0
            }
    },
    {
        "text": "Abu Dhabi is the capital of the United Arab Emirates, a country located in the Middle East. UAE has a population of 9,890,400 and an area of 83,600 km²",
        "metadata": {
            "name": "united arab emirates", 
            "capital": "abu dhabi", 
            "population": 9890400, 
            "area": 83600
            }
    }
    ]
    countries.extend(manual)
    
    return countries


def store_in_chroma(countries):
    # Initialize ChromaDB client and create collection
    client = chromadb.Client()
    collection = client.create_collection("countries", get_or_create=True)

    # Clear existing data in the collection
    collection.delete(where={"name": {"$ne": "__DOES_NOT_EXIST__"}})
    # collection.drop()  # Uncomment to drop the collection if needed
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
  
    # Add countries to the collection
    for country in countries:
        embedding = model.encode(country['text']).tolist()
        collection.add(
            documents=[country['text']],
            metadatas=[country['metadata']],
            embeddings=[embedding],
        # Generate a unique ID for each document to CHromaDB to avoid conflicts
            ids=[str(uuid.uuid4())]
        )

if __name__ == "__main__":
    # Scrape countries and store in ChromaDB
    countries = scrape_countries()
    store_in_chroma(countries)
    print(f"Stored {len(countries)} countries in ChromaDB")