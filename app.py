import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMNI_API_KEY"))

#Text Extraction
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                lines = [line.strip() for line in page_text.split("\n")if line.strip()] 
                text += " ".join(lines)+ " "
        
        return text
    except Exception as e:
        st.error(f"❌ Error while extracting text: {e}")
        return None
        
    

#Chunking
def chunking(text,chunk_size = 500):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = 150
    )
    chunks = splitter.split_text(text)
    return chunks

#Get embeddings from Gemini
def get_embeddings(texts):
    embeddings = []
    try:
        for t in texts:
            result = genai.embed_content(model="models/embedding-001", content=t)
            embeddings.append(result["embedding"])
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        st.error(f"❌ Error while generating embeddings: {e}")
        return None

#FAISS index generation
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

#Query generation for gemni
def query_gemini(user_query,chunks,index):
    try:
        q_embed = get_embeddings([user_query])
        if q_embed is None:
            return "Error generating query embedding."

        _, I = index.search(q_embed, k=3)
        context = "\n".join([chunks[i] for i in I[0]])

        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {user_query}\nAnswer:"
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error while querying Gemini: {e}"


# StreamLit UI

st.set_page_config(page_title="PDF Chatbot", page_icon= "attachment_129361955.jpeg")
st.title("PDF Chatbot(Gemini + FAISS)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
if uploaded_file is not None and "index" not in st.session_state:
    st.info("Processing PDF and building vector database...")
    text = extract_text_from_pdf(uploaded_file)
    if text:
        chunks = chunking(text)
        embeddings = get_embeddings(chunks)
        
        if embeddings is not None:
            st.session_state.chunks = chunks
            st.session_state.index = build_faiss_index(embeddings)
            st.success("PDF processed successfully! You can now ask questions. ✅")

        else:
            st.error("Failed to generate Embeddings. Try Again")
        
        

if "index" in st.session_state:
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        answer = query_gemini(user_query, st.session_state.chunks, st.session_state.index)
        st.markdown("### 🤖 Chatbot Answer:")
        st.write(answer)