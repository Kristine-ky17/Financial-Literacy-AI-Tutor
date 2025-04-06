# Helper functions will be stored in this file
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# def load_pdfs_from_folder(folder_path):
#     texts = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             with open(os.path.join(folder_path, filename), "rb") as f:
#                 reader = PyPDF2.PdfReader(f)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text() or ""
#                 texts.append(text)
#     return texts

def load_text_from_pdf(filename):
    texts = []
    with open(filename, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts

def chunk_texts(texts, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
