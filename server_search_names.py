from fastapi import FastAPI
from pymilvus import MilvusClient
from pymilvus import model
from utils.timer import timer
from utils.printing import printgreen, printred, printblue
from utils.embedddigs import MilvusSearch
import time

app = FastAPI()
print("FastAPI app created")
milvus_client = MilvusSearch(uri="http://localhost:19530")
print("Milvus client created")
milvus_client.LoadEmbeddingFunction("all-MiniLM-L6-v2", device="cpu")
print("Embedding function loaded")
from utils.convert_between_latin_and_cyril import identify_and_convert


@app.get("/search/{name}")
def search_name(name: str, Limit: int = 10):
    output = {
        "uzbek_names": [],
        "russian_names": [],
    }
    detected_lang, source_lang, converted_name = identify_and_convert(name)
    printblue(f"Detected language: {detected_lang}")
    printblue(f"Original name: {name}")
    printblue(f"Source language: {source_lang}")
    printblue(f"Converted name: {converted_name}")

    if detected_lang == "uz":
        starting_time = time.time()
        embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
        embedded_name_2 = milvus_client.create_embedding_for_word(name)
        printred(f"Time for embedding:{time.time() - starting_time} ")
        starting_time = time.time()
        res_1 = milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_2,
            filter="",
            limit=Limit,
            output_fields=["name"],
        )
        printred(f"Time taken for search uzbek: {time.time() - starting_time}")
        starting_time = time.time()
        res_2 = milvus_client.search(
            collection_name="rus_names",
            data=embedded_name_1,
            filter="",
            limit=Limit,
            output_fields=["name"],
        )
        printred(f"Time taken for search russian : {time.time() - starting_time}")

        # printgreen(" ".join([r["entity"]["name"] for r in res]))
    elif detected_lang == "ru":
        starting_time = time.time()
        embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
        embedded_name_2 = milvus_client.create_embedding_for_word(name)
        printred(f"Time for embedding:{time.time() - starting_time} ")
        starting_time = time.time()
        res_1 = milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_1,
            filter="",
            limit=Limit,
            output_fields=["name"],
        )
        printred(f"Time taken for search russian: {time.time() - starting_time}")
        starting_time = time.time()
        res_2 = milvus_client.search(
            collection_name="rus_names",
            data=embedded_name_2,
            filter="",
            limit=Limit,
            output_fields=["name"],
        )
        printred(f"Time taken for search uzbek: {time.time() - starting_time}")

    # printgreen(" ".join([r["entity"]["name"] for r in res]))
    for r in res_1[0]:
        output["uzbek_names"].append(
            {"name": r["entity"]["name"], "score": r["distance"]}
        )
    for r in res_2[0]:
        output["russian_names"].append(
            {"name": r["entity"]["name"], "score": r["distance"]}
        )
    return output


@app.get("/embed/{name}")
def embed_name(name: str):
    vectors = milvus_client.create_embedding_for_word(name)
    return vectors
