from fastapi import FastAPI, Request, Response
from pymilvus import MilvusClient
from pymilvus import model
from utils.timer import timer
from utils.printing import printgreen, printred, printblue
from utils.convert_between_latin_and_cyril import identify_and_convert
from utils.embedddigs import MilvusSearch
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import time
import dotenv
import os

dotenv.load_dotenv()
app = FastAPI()
print("FastAPI app created")
milvus_client = MilvusSearch(os.getenv("MILVUS_URI", "http://localhost:19530"))
print("Milvus client created")
milvus_client.LoadEmbeddingFunction("all-MiniLM-L6-v2", device="cpu")
print("Embedding function loaded")


# Prometheus a Counter metric
REQUEST_COUNT = Counter("app_requests_total", "Total number of requests")


@app.get("/search/{name_input}")
def search_name(request: Request, name_input: str):
    REQUEST_COUNT.inc()
    try:
        Limit = int(request.query_params.get("limit", 10))
    except Exception as e:
        print(e)
        Limit = 10
    output = {
        "uzbek_names": [],
        "russian_names": [],
    }

    names = name_input.split(" ")
    res_1 = []
    res_2 = []
    for name in names:

        detected_lang, source_lang, converted_name = identify_and_convert(name)
        printblue(f"Detected language: {detected_lang}")
        printblue(f"Original name: {name}")
        printblue(f"Source language: {source_lang}")
        printblue(f"Converted name: {converted_name}")
        if detected_lang == "uz":
            starting_time = time.time()
            embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
            embedded_name_2 = milvus_client.create_embedding_for_word(name)
            name_changed = name.replace("x", "h").replace("X", "H")
            embedded_name_3 = milvus_client.create_embedding_for_word(name_changed)
            printred(f"Time for embedding:{time.time() - starting_time} ")
            starting_time = time.time()

            res_1 += milvus_client.search(
                collection_name="uzbek_names",
                data=embedded_name_2,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            res_1 += milvus_client.search(
                collection_name="uzbek_names",
                data=embedded_name_3,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            printred(f"Time taken for search uzbek: {time.time() - starting_time}")
            starting_time = time.time()
            res_2 += milvus_client.search(
                collection_name="rus_names",
                data=embedded_name_1,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            output["uzbek_names"].append({"name": name, "score": 1})
            output["uzbek_names"].append({"name": name_changed, "score": 1})
            output["russian_names"].append({"name": converted_name, "score": 1})
            printred(f"Time taken for search russian : {time.time() - starting_time}")

            # printgreen(" ".join([r["entity"]["name"] for r in res]))
        elif detected_lang == "ru":
            starting_time = time.time()
            embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
            embedded_name_2 = milvus_client.create_embedding_for_word(name)
            name_changed = converted_name.replace("x", "h").replace("X", "H")
            embedded_name_3 = milvus_client.create_embedding_for_word(name_changed)
            printred(f"Time for embedding:{time.time() - starting_time} ")
            starting_time = time.time()
            res_1 += milvus_client.search(
                collection_name="uzbek_names",
                data=embedded_name_1,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            res_1 += milvus_client.search(
                collection_name="uzbek_names",
                data=embedded_name_3,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            printred(f"Time taken for search uzbek: {time.time() - starting_time}")
            starting_time = time.time()
            res_2 += milvus_client.search(
                collection_name="rus_names",
                data=embedded_name_2,
                filter="",
                limit=Limit,
                output_fields=["name"],
            )
            output["uzbek_names"].append({"name": converted_name, "score": 1})
            output["uzbek_names"].append({"name": name_changed, "score": 1})
            output["russian_names"].append({"name": name, "score": 1})

            printred(f"Time taken for search russian: {time.time() - starting_time}")
        print("Length of res_1: ", len(res_1))
        print("Length of res_2: ", len(res_2))
    # printgreen(" ".join([r["entity"]["name"] for r in res]))
    for i in range(len(res_1)):
        for r in res_1[i]:
            output["uzbek_names"].append(
                {"name": r["entity"]["name"], "score": r["distance"]}
            )
    for i in range(len(res_2)):
        for r in res_2[i]:
            output["russian_names"].append(
                {"name": r["entity"]["name"], "score": r["distance"]}
            )

    return output


@app.get("/embed/{name}")
def embed_name(name: str):
    REQUEST_COUNT.inc()
    vectors = milvus_client.create_embedding_for_word(name)
    return vectors


@app.get("/metrics")
def metrics():
    # Generate latest metrics for Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
