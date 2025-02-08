import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store():
    with open("extracted_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract text fields from each dictionary and check for the word "course"
    filtered_data = [
        f"{item['title']} {item['description']}" 
        for item in raw_data 
        if "course" in item.get("description", "").lower()
    ]

    # Ensure there is data to process
    if not filtered_data:
        print("No relevant data found for vector storage.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents(filtered_data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")

    print("Vector database updated with filtered data.")

if __name__ == "__main__":
    create_vector_store()
