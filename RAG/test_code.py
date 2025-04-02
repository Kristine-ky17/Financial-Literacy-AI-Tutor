from utils import *


texts = load_pdfs_from_folder("data")
print(len(texts)) #1
chunks = chunk_texts(texts)
print(len(chunks)) #64