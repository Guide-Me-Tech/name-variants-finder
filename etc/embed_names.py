filenames = [
    "uzname-men-name-lat-v1.1b 2.txt",
    "uzname-women-name-lat-v1.1b.txt",
    "russian_male_names.txt",
    "russian_female_names.txt",
]


import os
import json
import chromadb
from chromadb.utils import embedding_functions

uz_names = []
russian_names = []


with open(filenames[0], "r") as f:
    names = f.readlines()
    uz_names += [name.strip() for name in names]
with open(filenames[1], "r") as f:
    names = f.readlines()
    uz_names += [name.strip() for name in names]

with open(filenames[2], "r") as f:
    names = f.readlines()
    russian_names += [name.strip() for name in names]
with open(filenames[3], "r") as f:
    names = f.readlines()
    russian_names += [name.strip() for name in names]


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


for i in range(len(uz_names) // 5461):
    if 5461 * (i + 1) < len(uz_names):
        input = uz_names[i * 5461 : (i + 1) * 5461]
    else:
        input = uz_names[i * 5461 :]

    uz_names_collection.upsert(ids=[str(i) for i in range(len(input))], documents=input)
# for i in range(len(russian_names) // 5461):
#     if 5461 * (i + 1) < len(russian_names):
#         input = russian_names[i * 5461 : (i + 1) * 5461]
#     else:
#         input = russian_names[i * 5461 :]
input = russian_names
russian_names_collection.upsert(
    ids=[str(i) for i in range(len(input))], documents=input
)
