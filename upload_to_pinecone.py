import os
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load your keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")  # Not needed anymore but okay to leave

# Init Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "richardhayes-content"

# Check and create index if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Your article URLs
urls = [
    "https://www.richardhayes.co.nz/how-will-generative-ai-impact-search/",
    "https://www.richardhayes.co.nz/social-media-has-a-problem/",
    "https://www.richardhayes.co.nz/ai-2027/"
]

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_page_text(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join(p.get_text() for p in paragraphs)

# Process and upload
for url in urls:
    print(f"\nProcessing: {url}")
    text = get_page_text(url)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embed = openai.embeddings.create(
    input=chunk,
    model="text-embedding-ada-002"
).data[0].embedding

        index.upsert(vectors=[
            {
                "id": f"{url}--{i}",
                "values": embed,
                "metadata": {
                    "url": url,
                    "text": chunk
                }
            }
        ])
        print(f"Uploaded chunk {i + 1}/{len(chunks)}")
