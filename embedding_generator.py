import pandas as pd
import joblib
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
MODEL_DIR = 'model_assets_rag'
SCRAPED_DATA_PATH = os.path.join(MODEL_DIR, 'catalog_data.pkl')
VECTOR_STORE_PATH = os.path.join(MODEL_DIR, 'assessment_vector_store.pkl')

# Placeholder for the embedding dimensionality. 
# This should match the output of the chosen embedding model (e.g., 384 for MiniLM, 768 for larger models).
EMBEDDING_DIMENSION = 384 

def load_catalog_data(file_path: str) -> pd.DataFrame:
    """Loads the structured catalog data from the simulation step."""
    try:
        df = joblib.load(file_path)
        print(f"Loaded structured catalog data with {len(df)} assessments.")
        return df
    except Exception as e:
        print(f"FATAL ERROR: Could not load structured catalog data from {file_path}. Error: {e}")
        print("Please ensure you have run 'data_pipeline.py' successfully first.")
        return pd.DataFrame()

def get_embedding_text(row: pd.Series) -> str:
    """Combines key fields into a single text string for embedding."""
    # This text is what the model will turn into a vector.
    return (
        f"Assessment Name: {row['Assessment_name']}. "
        f"Test Type: {', '.join(row['Test_type'])}. "
        f"Description: {row['Description']}"
    )

def simulate_embedding_generation(text_list: List[str]) -> np.ndarray:
    """
    SIMULATED FUNCTION for embedding generation.
    
    In a real RAG system, this would call the Gemini Embedding API (or a local 
    Sentence Transformer model) to generate real, high-quality vectors.
    
    Example real code structure (using a hypothetical LLM client):
    
    from gemini_client import get_embeddings
    
    def real_embedding_generation(text_list):
        # Assumes get_embeddings returns a list of NumPy arrays
        embeddings = get_embeddings(model="gemini-embedding-001", texts=text_list)
        return np.array(embeddings)
    """
    print(f"Simulating embedding generation for {len(text_list)} documents...")
    # Generating random vectors for simulation purposes
    embeddings = np.random.rand(len(text_list), EMBEDDING_DIMENSION).astype(np.float32)
    # Normalize vectors (important for cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def create_vector_store(df_catalog: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generates embeddings and combines them with metadata to create the Vector Store.
    
    Returns: A tuple of (embeddings_matrix, metadata_list)
    """
    if df_catalog.empty:
        raise ValueError("Catalog DataFrame is empty. Cannot create vector store.")

    # 1. Prepare text for embedding
    embedding_texts = df_catalog.apply(get_embedding_text, axis=1).tolist()
    
    # 2. Generate Embeddings (Simulated)
    embeddings_matrix = simulate_embedding_generation(embedding_texts)
    
    # 3. Prepare Metadata (what we need to retrieve after search)
    # We strip the embedding text, as it's redundant once the vector is created.
    metadata_list = df_catalog[[
        'Assessment_url', 
        'Assessment_name', 
        'Test_type',
        'Description' # Keeping description for the final recommendation table
    ]].to_dict('records')
    
    print(f"Generated embeddings matrix of shape: {embeddings_matrix.shape}")
    
    return embeddings_matrix, metadata_list

def main():
    """Main function to execute the embedding generation and vector store creation."""
    
    Path(MODEL_DIR).mkdir(exist_ok=True)
    
    # 1. Load data from the scraping simulation
    df_catalog = load_catalog_data(SCRAPED_DATA_PATH)
    if df_catalog.empty:
        return
    
    # 2. Create the Vector Store
    try:
        embeddings_matrix, metadata_list = create_vector_store(df_catalog)
        
        vector_store = {
            'embeddings': embeddings_matrix,
            'metadata': metadata_list,
        }
        
        # 3. Save the Vector Store
        joblib.dump(vector_store, VECTOR_STORE_PATH)
        print(f"\nVector Store successfully saved to {VECTOR_STORE_PATH}")
        print("Vector Store is now ready for the RAG API service.")
        
    except ValueError as e:
        print(f"Error during vector store creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()