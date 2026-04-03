from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load FAQ data
loader = TextLoader("data/faq.txt")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create embeddings using Ollama (FREE)
embeddings = OllamaEmbeddings(model="phi3")

# Store in ChromaDB
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db"
)

db.persist()

print("✅ Knowledge base created!")