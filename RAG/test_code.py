from utils import *
from langchain.schema import Document


documents = list()
for file in os.listdir("data/insurance"):
    all_texts = load_text_from_pdf("data/insurance/" + file)
    all_text_chunks = chunk_texts(all_texts)
    # print(all_text_chunks)
    documents.extend([
        Document(page_content=chunk, metadata={"source": file, "content": "insurance"}) 
        for chunk in all_text_chunks
    ])
print(len(documents))