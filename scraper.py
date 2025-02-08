from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def extract_data(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Extract only meaningful text using BeautifulSoup
    cleaned_texts = []
    for doc in documents:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)  # Remove HTML tags
        cleaned_texts.append(text)

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.create_documents(cleaned_texts)

    return chunks

if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    data = extract_data(url)

    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump([chunk.page_content for chunk in data], f, indent=4)

    print("Data extraction complete!")
