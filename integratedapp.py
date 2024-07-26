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
from io import BytesIO
from docx import Document
from pptx import Presentation
import csv
from txtai.embeddings import Embeddings

load_dotenv()

# Configure the API key
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

# Configure the chat model
model = genai.GenerativeModel(
    model_name='gemini-pro',
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    })
chat = model.start_chat(history=[])

# Initialize txtai embeddings
txtai_embeddings = Embeddings({"path": "sentence-transformers/paraphrase-MiniLM-L3-v2", "content": True})

# Function to read all PDFs from a folder and return concatenated text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to read DOCX files and return concatenated text
def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Function to read PPTX files and return concatenated text
def get_pptx_text(pptx_docs):
    text = ""
    for pptx in pptx_docs:
        presentation = Presentation(pptx)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    return text

# Function to read TXT files and return concatenated text
def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode('utf-8') + "\n"
    return text

# Function to read CSV files and return concatenated text
def get_csv_text(csv_docs):
    text = ""
    for csvfile in csv_docs:
        csvreader = csv.reader(csvfile.read().decode('utf-8').splitlines())
        for row in csvreader:
            text += ' '.join(row) + "\n"
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Get txtai embeddings
def create_txtai_embeddings(text_chunks):
    data = [{"text": chunk} for chunk in text_chunks]
    txtai_embeddings.index(data)

# Function to search using txtai
def search_txtai(query):
    results = txtai_embeddings.search(query, limit=10)
    return results

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "You can ask questions about the provided documents."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    context = "\n".join([doc.page_content for doc in docs])
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)

    return response['output_text']

def summarize_text(text):
    summary_prompt = PromptTemplate(
        template="Summarize the following text: {input_text}",
        input_variables=["input_text"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.5)
    summary_chain = load_qa_chain(llm=model, chain_type="stuff", prompt=summary_prompt)
    summary = summary_chain.run({"input_text": text})
    return summary['output_text']

def main():
    st.set_page_config(page_title="NYP Chatbot", page_icon="🖐", layout="wide")

    st.title("Upload and Process Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your files (PDF, DOCX, PPTX, TXT, CSV)", accept_multiple_files=True,
        type=['pdf', 'docx', 'pptx', 'txt', 'csv'])

    if uploaded_files:
        all_text = ""
        pdf_files = [file for file in uploaded_files if file.name.endswith(".pdf")]
        docx_files = [file for file in uploaded_files if file.name.endswith(".docx")]
        pptx_files = [file for file in uploaded_files if file.name.endswith(".pptx")]
        txt_files = [file for file in uploaded_files if file.name.endswith(".txt")]
        csv_files = [file for file in uploaded_files if file.name.endswith(".csv")]

        if pdf_files:
            all_text += get_pdf_text(pdf_files)
        if docx_files:
            all_text += get_docx_text(docx_files)
        if pptx_files:
            all_text += get_pptx_text(pptx_files)
        if txt_files:
            all_text += get_txt_text(txt_files)
        if csv_files:
            all_text += get_csv_text(csv_files)

    st.title("Chat with PDF files using Gemini 🙋‍♂")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "You can ask questions about the provided documents."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['role'].capitalize()}: {message['content']}")

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"User: {prompt}")

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(f"Assistant: {response}")

    st.title("Search and View Content")
    search_query = st.text_input("Enter search query")
    if st.button("Search"):
        if search_query:
            results = search_txtai(search_query)
            st.write("Search Results:")
            for result in results:
                st.write(result)
        else:
            st.write("Please enter a query to search.")

    if st.button("View Content"):
        if uploaded_files:
            st.write("Content of the uploaded files:")
            st.write(all_text)
        else:
            st.write("No files uploaded.")

if __name__ == "__main__":
    main()
