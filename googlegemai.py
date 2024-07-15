import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

# Configure the API key
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    model_name='gemini-pro',
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    })
chat = model.start_chat(history=[])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_files):
    text = ""
    for txt_file in txt_files:
        text += txt_file.read().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to handle user input
def handle_user_input():
    user_question = st.session_state.user_input
    if user_question:
        try:
            response = chat.send_message(user_question)
            st.session_state.history.append(("You", user_question))
            st.session_state.history.append(("NYP Chatbot", response.text))
            # Clear the input field
            st.session_state.user_input = ""
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Streamlit App
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("NYP CNC Chatbot")

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for sender, message in st.session_state.history:
            st.markdown(f"**{sender}:**")
            st.markdown(f"{message}")

    # Fixed input field at the bottom
    st.text_input("You: ", key="user_input", on_change=handle_user_input)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF or TXT Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]
                    txt_files = [f for f in uploaded_files if f.type == "text/plain"]
                    raw_text = get_pdf_text(pdf_files) + get_txt_text(txt_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
