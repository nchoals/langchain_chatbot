from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Function to read text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to break text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to convert chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Test on an individual PDF file
pdf_file = "documentation/the-tortoise-and-the-hare-story.pdf"

# Step 1: Read text from the PDF file
pdf_text = get_pdf_text([pdf_file])
print("PDF Text:", pdf_text[:500])  # Print the first 500 characters for verification

# Step 2: Split the text into chunks
text_chunks = get_text_chunks(pdf_text)
print("Number of text chunks:", len(text_chunks))
print("First text chunk:", text_chunks[0])  # Print the first chunk for verification

# Step 3: Convert the text chunks into vectors and save the FAISS index
get_vector_store(text_chunks)

print("FAISS index has been created and saved locally.")
