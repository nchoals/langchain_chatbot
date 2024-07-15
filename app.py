import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure the API key
API_KEY = "AIzaSyAJ01POk5Rh3Alss8r0faESxvL2Ecr-YtI"

genai.configure(
    api_key=API_KEY
)

# Function to read text from PDF
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#break text into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#convert chunks into vectors
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=API_KEY)
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

# ------------------------ TEST FUNCTIONS ---------------------------------
# #Test on an individual PDF file
# pdf_file = "documentation/Bible.pdf"

# #Read text from the PDF file
# pdf_text = get_pdf_text([pdf_file])
# # print("PDF Text:", pdf_text[:200])  # Print the first 500 characters for verification

# #Split the text into chunks
# text_chunks = get_text_chunks(pdf_text)
# print("Number of text chunks:", len(text_chunks))
# print("First text chunk:", text_chunks[0])  # Print the first chunk for verification

# #Convert the text chunks into vectors and save the FAISS index
# get_vector_store(text_chunks)
# print("FAISS index has been created and saved locally.")

# Function to load and inspect the FAISS index
# def load_and_inspect_faiss_index(index_path):
#     vector_store = FAISS.load_local(index_path, embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
#     # Get the first vector and its corresponding text chunk
#     vectors, _ = vector_store.index.reconstruct_n(0, 5)  # Load the first 5 vectors for inspection
#     return vectors

# #Load and inspect the FAISS index
# vectors = load_and_inspect_faiss_index("faiss_index")
# print("First 5 vectors:", vectors)

# ---------------------------------------------------------------------------

def get_conversational_chain():
    prompt_template="""
    Answer the questions as detailed as possible from provided context, if context not available
    say, "Answer is not available in this context." Dont provide the wrong answer.
    Context: \n {context}? \n
    Question: \n {question} ? \n

    Answer: 
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key=API_KEY)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain
                   
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=API_KEY)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True
    )

    print(response)
    st.write("reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("NYP Chatbot using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
