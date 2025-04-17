import streamlit as st
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from langchain.chains import RetrievalQA
import json
from pinecone import Pinecone

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

topics = [
    "economics",
    "financial markets",
    "foreign exchange",
    "fundraising",
    "insurance",
    "international trade and finance",
    "investing",
    "m&a",
    "real_estate",
    "wealth management",
    "corporate finance"
]


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "finance"

index = pc.Index(index_name)

# Load metadata lookup
with open("news_sources.json") as f:
    metadata_lookup = json.load(f)



def answer_qn(namespace: str, question: str):
    vectorstore = PineconeVectorStore(
        index=index, embedding=OpenAIEmbeddings(), namespace=namespace
    )

    retriever = vectorstore.as_retriever(search_kwargs = {"k":3})
    prompt_str = (
        "Answer the question below using the context:\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = RunnableParallel({
        "context":  retriever,
        "question": RunnablePassthrough(),
    }) | prompt | ChatOpenAI(
        model_name = "gpt-4o",
        temperature = 0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    answer = chain.invoke(question).content

    st.subheader("Answer:")
    st.write(answer)

    docs = retriever.invoke(user_question)
    docs = set([doc.metadata.get("source") for doc in docs])

    st.subheader("Sources used:")
    for source_key in docs:
        # source_key = doc.metadata.get("source")
        st.write(f"- {metadata_lookup.get(source_key, source_key)}")

    # llm = ChatOpenAI(
    #     model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    # )

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    #     return_source_documents=True,
    # )

    # response = qa_chain({"query": question})

    # if user_question:
    #     with st.spinner("Searching for the answer..."):
    #         response = qa_chain({"query": user_question})
    #         st.subheader("Answer:")
    #         st.write(response["result"])

    #         st.subheader("Sources used:")
    #         for doc in response["source_documents"]:
    #             st.write(f"- {metadata_lookup.get(doc.metadata['source'], doc.metadata['source'])}")


    # print("Answer:")
    # print(response["result"])

    # print("\n Sources used:")
    # for doc in response["source_documents"]:
    #     # print(f"- {doc.metadata['source']}")
    #     print(f"- {metadata_lookup.get(doc.metadata['source'])}")


st.title("Finance Document Q&A")
topics = [
    "economics",
    "financial markets",
    "foreign exchange",
    "fundraising",
    "insurance",
    "international trade and finance",
    "investing",
    "m&a",
    "real_estate",
    "wealth management",
    "corporate finance"
]

selected_topic = st.selectbox(
    "Select a finance topic:",
    topics,
    index=None,
    placeholder="Choose a topic..."
)

user_question = st.text_input("Enter your question:")

if selected_topic and user_question:
    answer_qn(selected_topic, user_question)
