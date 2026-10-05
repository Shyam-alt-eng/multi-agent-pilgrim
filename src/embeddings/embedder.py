"""Embedding model configuration used by ingestion and retrieval."""

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "intfloat/multilingual-e5-base"


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 wrapper applying the required passage/query prefixes."""

    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def create_embeddings():
    return E5Embeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
