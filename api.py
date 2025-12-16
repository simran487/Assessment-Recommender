import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
MODEL_DIR = 'model_assets_rag'
VECTOR_STORE_PATH = os.path.join(MODEL_DIR, 'assessment_vector_store.pkl')
EMBEDDING_DIMENSION = 384 # Must match the dimension in embedding_generator.py

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Assessment Recommendation API",
    description="A RAG-based API to recommend assessments based on user queries.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
# This is the key change to allow your frontend to communicate with the backend.
origins = [
    "http://localhost",
    "http://localhost:5173", # Default Vite dev server port
    "http://localhost:3000", # Default Create React App dev server port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Data Models for API ---
class Query(BaseModel):
    user_query: str

class Recommendation(BaseModel):
    Assessment_name: str
    Assessment_url: str
    Test_type: List[str]

# --- Load Vector Store at Startup ---
vector_store: Dict[str, Any] = {}

@app.on_event("startup")
def load_model():
    """Load the vector store from disk when the API starts."""
    global vector_store
    if not os.path.exists(VECTOR_STORE_PATH):
        raise RuntimeError(f"Vector store not found at {VECTOR_STORE_PATH}. Please run embedding_generator.py first.")
    vector_store = joblib.load(VECTOR_STORE_PATH)
    print("Vector store loaded successfully.")

# --- SIMULATED Functions (to be replaced with real logic) ---

def simulate_embedding_generation(text_list: List[str]) -> np.ndarray:
    """Simulates generating an embedding for the user query."""
    print(f"Simulating embedding for query: '{text_list[0]}'")
    query_embedding = np.random.rand(1, EMBEDDING_DIMENSION).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)
    return query_embedding

def find_top_k_similar(query_embedding: np.ndarray, doc_embeddings: np.ndarray, metadata: List[Dict], k: int = 5) -> List[Dict]:
    """
    Finds the top K most similar documents to a query embedding.

    Args:
        query_embedding: The embedding vector of the user's query.
        doc_embeddings: The matrix of all document embeddings.
        metadata: The list of metadata corresponding to each document.
        k: The number of top results to return.

    Returns:
        A list of the top K metadata dictionaries, sorted by similarity.
    """
    # Calculate cosine similarity between the query and all documents
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

    # Get the indices of the top K most similar documents
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Return the metadata for the top K documents
    return [metadata[i] for i in top_k_indices]

# --- Health Check Endpoint ---

@app.get("/health", status_code=200)
def health_check():
    """
    Health check endpoint to verify that the service is running
    and the vector store is loaded.
    """
    return {
        "status": "ok",
        "vector_store_loaded": bool(vector_store and 'embeddings' in vector_store)
    }

# --- API Endpoint ---

@app.post("/recommend", response_model=List[Recommendation])
def recommend_assessments(query: Query):
    """Receives a user query, finds relevant assessments, and returns them."""
    print(f"Received query: {query.user_query}")

    # 1. Get embeddings and metadata from the loaded vector store
    doc_embeddings = vector_store.get('embeddings')
    metadata = vector_store.get('metadata')

    if doc_embeddings is None or metadata is None:
        raise HTTPException(status_code=500, detail="Vector store is not properly loaded or is corrupted.")

    # 2. Generate an embedding for the incoming user query (simulated)
    query_embedding = simulate_embedding_generation([query.user_query])

    # 3. Perform the vector search to find the top 5 recommendations
    recommendations = find_top_k_similar(query_embedding, doc_embeddings, metadata, k=5)

    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found. Vector store might be empty.")

    return recommendations