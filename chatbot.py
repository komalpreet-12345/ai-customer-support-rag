from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local LLM (FREE)
llm = Ollama(model="llama3")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever()

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def ask_bot(question):
    result = qa_chain({"question": question})
    return result["answer"]