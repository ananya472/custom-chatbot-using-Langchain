# custom-chatbot-using-Langchain

LangChain Chatbot Overview

This chatbot project scrapes course data from Brainlox, processes it into embeddings using a Hugging Face model, and stores it in a FAISS vector database for quick retrieval.

app.py: A Flask REST API that handles user queries and returns the most relevant course info.
scraper.py: Scrapes course data from Brainlox and saves it to extracted_data.json.
vector_store.py: Converts scraped data into embeddings and stores them in FAISS (index.faiss and index.pkl).

How it Works:

Scraping: scraper.py pulls course details from the web.
Embedding: vector_store.py uses Hugging Face models to create embeddings, storing them in FAISS for fast similarity search.
Querying: app.py takes user queries, searches the vector store, and returns the best-matching courses.
