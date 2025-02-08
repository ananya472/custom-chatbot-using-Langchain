import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# FAISS index path
DB_FAISS_PATH = r"C:\Users\anany\Desktop\custom-chatbot\faiss_index"

# Caching FAISS index to avoid reloading on every query
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS: {str(e)}")
        return None

# Custom prompt template for structured responses
def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are an AI assistant trained on company-specific data. Your goal is to provide structured responses to user queries.

        If the relevant information is not found, respond with: "I'm sorry, but I couldn't find relevant information."

        Context: {context}
        Question: {question}

        Respond in the following format:
        - **Course Name**: [Title]
        - **Description**: [Brief description of the course]
        - **Lessons**: [Number of lessons]
        - **Price per session**: [Course fee]
        """,
        input_variables=["context", "question"]
    )

# Load Hugging Face LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.1,  # Lower randomness for factual responses
        top_p=0.85,  # Slightly more controlled nucleus sampling
        repetition_penalty=1.3,  # Reduce redundant outputs
        max_length=250,  # Allow longer responses if needed
        token=HF_TOKEN
    )

# Streamlit UI
def main():
    st.title("Custom Chatbot (Trained on Your Data)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User query input
    user_query = st.chat_input("Ask me anything about courses...")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Load credentials
        HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B" 
        HF_TOKEN = os.getenv("HF_TOKEN")

        if not HF_TOKEN:
            st.error("Hugging Face token is missing! Set it in your environment variables.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load FAISS vector store.")
                return

            # Retrieve relevant documents with better filtering
            docs = vectorstore.similarity_search(user_query, k=5)  # Fetch more results

            if not docs:
                st.chat_message("assistant").markdown("I'm sorry, but I couldn't find relevant information.")
                return

            # Initialize QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.7}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            # Generate response
            response = qa_chain.run(user_query)

            # Ensure structured response formatting
            if isinstance(response, str):
                response = response.replace("\n", " ").strip()

            st.chat_message("assistant").markdown(f"**Response:** {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Run Streamlit app
if __name__ == "__main__":
    main()

