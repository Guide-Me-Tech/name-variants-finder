import chromadb
from chromadb.utils import embedding_functions


stef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_instance = chromadb.PersistentClient("./names")

uz_names_collection = chroma_instance.get_or_create_collection(
    "uz_names", embedding_function=stef
)
russian_names_collection = chroma_instance.get_or_create_collection(
    "russian_names", embedding_function=stef
)


name_query = input("Enter a name: ")

uz_names = uz_names_collection.query(query_texts=[name_query], n_results=10)
russian_names = russian_names_collection.query(query_texts=[name_query], n_results=10)

print(uz_names.items())
print(russian_names.items())
