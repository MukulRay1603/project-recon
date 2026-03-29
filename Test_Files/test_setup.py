from dotenv import load_dotenv
import os

load_dotenv()

print("=== Testing imports ===")
import langgraph; print("✓ langgraph")
import langchain; print(f"✓ langchain {langchain.__version__}")
import gradio; print(f"✓ gradio {gradio.__version__}")
import semanticscholar; print("✓ semanticscholar")
from sentence_transformers import SentenceTransformer; print("✓ sentence-transformers")
import networkx; print(f"✓ networkx {networkx.__version__}")

print("\n=== Testing API keys ===")
groq_key = os.getenv("GROQ_API_KEY")
s2_key = os.getenv("S2_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
print(f"✓ GROQ_API_KEY: {'set' if groq_key else 'MISSING'}")
print(f"✓ S2_API_KEY: {'set — will activate in 1-3 days' if s2_key else 'not set yet (pending)' }")
print(f"✓ TAVILY_API_KEY: {'set' if tavily_key else 'MISSING'}")

print("\n=== Testing Groq connection ===")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
response = llm.invoke("Say exactly: setup confirmed")
print(f"✓ Groq response: {response.content}")

print("\n✅ Phase 1 complete — all systems go")