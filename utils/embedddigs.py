from pymilvus import MilvusClient
from pymilvus import model
import time


def PrepareData(names, embeddings) -> dict:
    data = []
    for i in range(len(names)):
        data.append({"id": i, "vector": embeddings[i], "name": names[i]})
    return data


class MilvusSearch(MilvusClient):
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: float | None = None,
        **kwargs
    ) -> None:
        super().__init__(uri, user, password, db_name, token, timeout, **kwargs)
        self.sentence_transformer_ef = None

    def insert_names(self, names, collection_name, dimension=384):
        if self.has_collection(collection_name=collection_name):
            self.drop_collection(collection_name=collection_name)
        else:
            self.create_collection(
                collection_name=collection_name,
                dimension=384,  # The vectors we will use in this demo has 768 dimensions
            )

    def LoadEmbeddingFunction(self, model_name, device="cpu"):
        try:
            self.sentence_transformer_ef = (
                model.dense.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,  # Specify the model name
                    device=device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                )
            )
        except Exception as e:
            print(e)

    def create_embedding_for_word(self, word) -> list:
        vectors = self.sentence_transformer_ef.encode_documents([word])
        return vectors
