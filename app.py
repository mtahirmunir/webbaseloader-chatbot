import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

# Load secrets for Streamlit Cloud
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", "default_project")

# Set environment variables (if needed for LangSmith tracking)
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Streamlit app UI
st.title("Document Retrieval & Query App")

# URL input for data ingestion
url = st.text_input("Enter the URL of the document to ingest:", 
                    "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")

# Button to load and process documents
if st.button("Load Document"):
    try:
        # Load the document
        st.info("Loading document...")
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        st.success(f"Loaded and split document into {len(documents)} chunks.")
        
        # Create vector embeddings and store them in a FAISS vector database
        st.info("Creating vector embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstoredb = FAISS.from_documents(documents, embeddings)
        st.success("Document embeddings stored in vector database.")
        
        # Initialize retrieval mechanism
        retriever = vectorstoredb.as_retriever()
        
        # Set up LLM and chains
        llm = ChatOpenAI(model="gpt-4")
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        st.success("Ready to query the document!")
        
        # Query input and response
        query = st.text_input("Enter your query:")
        if st.button("Submit Query") and query.strip():
            st.info("Processing your query...")
            response = retrieval_chain.invoke({"input": query})
            answer = response['answer']
            context = response['context']
            
            # Display results
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Context:")
            st.write(context)
    except Exception as e:
        st.error(f"An error occurred: {e}")
