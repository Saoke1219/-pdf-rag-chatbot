from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

texts = ["Zizi Afrique is an organization focused on education.", "They produce annual reports."]

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_texts(texts=texts, embedding=embedding)

docs = db.similarity_search("What is Zizi Afrique?")
print(docs)


