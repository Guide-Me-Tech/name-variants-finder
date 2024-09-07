from sentence_transformers import SentenceTransformer

filenames = [
    "uzname-men-name-lat-v1.1b 2.txt",
    "uzname-women-name-lat-v1.1b.txt",
    "russian_male_names.txt",
    "russian_female_names.txt",
]
model = SentenceTransformer("all-MiniLM-L6-v2")

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


texts = russian_names
embeddings = model.encode(texts)  # Get the embeddings as numpy arrays


import redis
import numpy as np

# Connect to Redis
r = redis.Redis(host="localhost", port=6379)

# Store embeddings
for i, embedding in enumerate(embeddings):
    key = f"doc:{i}"
    r.hset(
        key,
        mapping={
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes()  # Store as binary
        },
    )
