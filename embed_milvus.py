from pymilvus import model

sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # Specify the model name
    device="cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)


filenames = [
    "data_names/uzbek_names_set_merged.txt",
    "data_names/russian_names_set_merged.txt",
]
uz_names = []
russian_names = []
with open(filenames[0], "r") as f:
    names = f.readlines()
    uz_names += [name.strip() for name in names]


with open(filenames[1], "r") as f:
    names = f.readlines()
    russian_names += [name.strip() for name in names]


docs_embeddings_russian = sentence_transformer_ef.encode_documents(russian_names)
docs_embeddings_uzbek = sentence_transformer_ef.encode_documents(uz_names)

# print("Embeddings:", docs_embeddings)
print("Dim:", sentence_transformer_ef.dim, docs_embeddings_russian[0].shape)
print("Dim:", sentence_transformer_ef.dim, docs_embeddings_uzbek[0].shape)


# print("Embeddings:", docs_embeddings)
# print("Dim:", sentence_transformer_ef.dim, docs_embeddings[0].shape)
import os

from pymilvus import MilvusClient

client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

if client.has_collection(collection_name="rus_names"):
    client.drop_collection(collection_name="rus_names")
client.create_collection(
    collection_name="rus_names",
    dimension=384,  # The vectors we will use in this demo has 768 dimensions
)

if client.has_collection(collection_name="uzbek_names"):
    client.drop_collection(collection_name="uzbek_names")
client.create_collection(
    collection_name="uzbek_names",
    dimension=384,  # The vectors we will use in this demo has 768 dimensions
)


def prepare_data(names, embeddings) -> dict:
    data = []

    for i in range(len(names)):
        data.append({"id": i, "vector": embeddings[i], "name": names[i]})
    return data


client.insert(
    collection_name="rus_names",
    data=prepare_data(russian_names, docs_embeddings_russian),
)
client.insert(
    collection_name="uzbek_names", data=prepare_data(uz_names, docs_embeddings_uzbek)
)
