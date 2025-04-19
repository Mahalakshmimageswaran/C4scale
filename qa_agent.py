import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import nltk
from nltk.corpus import stopwords

# --- Configuration ---
MAX_LINKS = 50
VISITED_URLS = set()

# Download NLTK resources
nltk.download('stopwords')

# --- Helper Functions ---
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_base_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def scrape_and_process(url, base_url):
    if url in VISITED_URLS or len(VISITED_URLS) >= MAX_LINKS or not url.startswith(base_url):
        return None, []

    try:
        print(f"Scraping: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        VISITED_URLS.add(url)

        main_content = soup.find('main') or soup.find('article') or soup.body
        text_content = ""
        if main_content:
            tags_to_extract = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for tag in tags_to_extract:
                if tag.name == 'a' and tag.get('href'):
                    continue
                text_content += tag.get_text(separator='\n', strip=True) + "\n"

            # Filter out irrelevant or too short text segments
            text_content = [t for t in text_content.split("\n") if len(t) > 50]
            text_content = "\n".join(text_content)

        text_content = re.sub(r'\s+', ' ', text_content).strip()
        return text_content, [
            urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)
            if is_valid_url(urljoin(url, link.get('href')))
        ]

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None, []

def crawl_website(start_url):
    base_url = get_base_url(start_url)
    queue = [start_url]
    all_texts = []

    while queue and len(VISITED_URLS) < MAX_LINKS:
        current_url = queue.pop(0)
        text, links = scrape_and_process(current_url, base_url)

        if text:
            all_texts.append((text, current_url))

        for link in links:
            if link.startswith(base_url) and link not in VISITED_URLS and link not in queue:
                queue.append(link)

    return all_texts

def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [{"page_content": text, "source": url} for text, url in texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.create_documents(
        [doc["page_content"] for doc in documents],
        metadatas=[{"source": doc["source"]} for doc in documents]
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def preprocess_query(query):
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r'\b\w+\b', query.lower())
    return ' '.join(word for word in tokens if word not in stop_words)

def semantic_search_agent():
    """Main function to run the semantic search agent."""
    print("Enter the help website URL: ")
    url = input().strip()

    print(f"Crawling and processing website: {url}...")
    texts = crawl_website(url)

    if not texts:
        print("Could not process the website or found no content.")
        return

    print("Creating vectorstore for efficient querying...")
    try:
        vectorstore = create_vectorstore(texts)
        print("Vectorstore created successfully.")

        DISTANCE_THRESHOLD = 1.5  # Increased the threshold

        while True:
            print("\nAsk a question (or type 'exit' to quit):")
            question = input().strip()

            if question.lower() == 'exit':
                break

            processed_question = preprocess_query(question)
            try:
                relevant_docs_with_scores = vectorstore.similarity_search_with_score(processed_question, k=5)

                print(f"Question: {question}")
                if relevant_docs_with_scores:
                    top_distance = relevant_docs_with_scores[0][1]
                    print(f"Top document distance: {top_distance:.4f}")
                    if top_distance <= DISTANCE_THRESHOLD:
                        print("Top relevant documents:")
                        for doc, score in relevant_docs_with_scores:
                            print(f"  - '{doc.page_content[:200]}...' (Distance: {score:.4f}, Source: {doc.metadata['source']})")
                    else:
                        print("No relevant information found for the query.")
                else:
                    print("No information found for the query.")

            except Exception as e:
                print(f"Error during querying: {e}")

    except Exception as e:
        print(f"Error creating vectorstore or during querying: {e}")

if __name__ == "__main__":
    semantic_search_agent()