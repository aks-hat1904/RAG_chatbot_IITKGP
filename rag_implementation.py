import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def get_rag_response(query):

    prompt_template = """Use the following context to answer the question.
    If you don't know the answer, say you don't know.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # "stuff" is the default; alternatives: "map_reduce", "refine", "map_rerank"
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    result = qa_chain({"query": query})
    return result["result"]

