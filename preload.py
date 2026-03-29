# Forces model download during build so startup is instant
from sentence_transformers import SentenceTransformer
print("Downloading all-MiniLM-L6-v2...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")