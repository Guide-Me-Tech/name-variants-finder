# Encode the query text
from sentence_transformers import SentenceTransformer
import numpy as np
import redis

model = SentenceTransformer("all-MiniLM-L6-v2")
q = input("Enter search text: ")
query_embedding = model.encode([q])[0]

# Search for similar vectors
r = redis.Redis(host="localhost", port=6379)

result = r.ft("embeddings_idx").search(
    query="*=>[KNN 3 @embedding $query_embedding AS score]",
    query_params={
        "query_embedding": np.array(query_embedding, dtype=np.float32).tobytes()
    },
)
