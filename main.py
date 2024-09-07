import os
import json
import chromadb
from chromadb.utils import embedding_functions

uz_names = []
russian_names = []


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


uz_names_collection.upsert(
    ids=[str(i) for i in range(len(uz_names))], documents=uz_names
)
russian_names_collection.upsert(
    ids=[str(i) for i in range(len(russian_names))], documents=russian_names
)
