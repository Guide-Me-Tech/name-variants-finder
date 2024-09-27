from pymilvus import MilvusClient
from pymilvus import model
from dotenv import load_dotenv

load_dotenv()
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # Specify the model name
    device="cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)
client = MilvusClient(uri="http://localhost:19530")

import time

starting_time = time.time()


def create_embedding_for_word(word):
    vectors = sentence_transformer_ef.encode_documents([word])
    return vectors


# print(create_embedding_for_word("Roman"))
embedded_name = create_embedding_for_word("роман")
print("Time for embedding: ", time.time() - starting_time)
starting_time = time.time()
res = client.search(
    collection_name="rus_names",
    data=embedded_name,
    filter="",
    limit=10,
    output_fields=["name"],
)
print("Time taken for search: ", time.time() - starting_time)


print(res)
