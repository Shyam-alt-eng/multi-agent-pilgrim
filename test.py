from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU is available
    encode_kwargs={
        "normalize_embeddings": True
    },  # Recommended for cosine similarity
)

# Example verification
text_en = "passage: Birla Mandir is located in Hyderabad."
text_te = "passage: బిర్లా మందిర్ హైదరాబాద్ లో ఉంది."

vec_en = embeddings.embed_query(text_en)
vec_te = embeddings.embed_query(text_te)

print(f"Embedding dimensions: {len(vec_en)}")