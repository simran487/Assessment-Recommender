import pandas as pd
import requests
import json
import os
import time
from typing import List, Dict, Any, Optional

# --- Configuration ---
# NOTE: Update this URL if your FastAPI server runs on a different address/port
API_BASE_URL = 'http://127.0.0.1:8000' 
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
# Updated to match the name of the test set file you uploaded
TEST_FILE_NAME = 'Gen_AI_Dataset_Train.csv' 
OUTPUT_FILE_NAME = 'submission_rag.csv'
MAX_RETRIES = 5
INITIAL_DELAY_SECONDS = 2 # Initial delay for exponential backoff

# --- Helper Functions ---

def load_csv_with_encoding(file_path: str) -> Optional[pd.DataFrame]:
    """Attempts to load a CSV file using common encodings to handle decode errors."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded file using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            # Continue to the next encoding if decode fails
            continue
        except Exception as e:
            print(f"Error reading file with encoding {encoding}: {e}")
            return None
            
    print(f"FATAL ERROR: Could not read CSV file '{file_path}' with any of the attempted encodings.")
    return None

def load_test_queries(file_path: str) -> Optional[List[str]]:
    """Loads unique queries from the test dataset."""
    
    # Use the new robust loading function
    df = load_csv_with_encoding(file_path)
    if df is None:
        return None
        
    try:
        # Determine the correct column name, looking for 'Query'
        query_col = next((col for col in df.columns if 'Query' in col), None)
        
        if not query_col:
            print(f"Error: Could not find 'Query' column in {file_path}. Columns found: {df.columns.tolist()}")
            return None
            
        queries = df[query_col].dropna().unique().tolist()
        print(f"Loaded {len(queries)} unique test queries from {file_path}")
        return queries
    except Exception as e:
        print(f"Error processing loaded queries: {e}")
        return None

def call_recommendation_api(query: str, max_retries: int = MAX_RETRIES) -> Optional[List[Dict[str, Any]]]:
    """
    Calls the FastAPI recommendation endpoint with exponential backoff.
    
    Args:
        query: The job description query string.
        max_retries: The maximum number of retries to attempt.
    
    Returns:
        A list of recommendation dictionaries, or None on failure.
    """
    payload = {"user_query": query}
    
    for attempt in range(max_retries):
        delay = INITIAL_DELAY_SECONDS * (2 ** attempt)
        if attempt > 0:
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)
            
        try:
            response = requests.post(
                RECOMMEND_ENDPOINT,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30 # Set a generous timeout
            )
            
            # Check for HTTP errors (4xx or 5xx)
            if response.status_code == 200:
                return response.json()
            else:
                # --- ENHANCED LOGGING HERE ---
                print(f"  Attempt {attempt + 1}: API returned HTTP {response.status_code}.")
                # Print the entire error response body for debugging server issues
                print(f"  Server Response Error Detail: {response.text}") 
                # If it's a 404 or a non-recoverable client error, we might stop sooner, 
                # but for robustness, we retry up to MAX_RETRIES.
                
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1}: Request failed (Connection Error): {e}")

    print(f"Failed to get recommendation for query after {max_retries} attempts.")
    return None

def generate_submission(queries: List[str], output_path: str):
    """
    Generates the final submission CSV by iterating through all test queries,
    saves the CSV, and outputs a preview table to the console.
    """
    results = []
    total_queries = len(queries)
    
    print(f"\n--- Starting Submission Generation ({total_queries} Queries) ---")
    
    for i, query in enumerate(queries):
        # Truncate the query for cleaner console output
        display_query = query[:70] + ('...' if len(query) > 70 else '')
        print(f"Processing Query {i+1}/{total_queries}: '{display_query}'")
        
        # 1. Get Recommendations from API
        recommendations = call_recommendation_api(query)
        
        if recommendations:
            # 2. Extract and format the results exactly as required: 
            # One row for each (Query, Assessment_url) pair.
            for rec in recommendations:
                results.append({
                    'Query': query,
                    'Assessment_url': rec['Assessment_url']
                })
        else:
            print(f"  WARNING: No recommendations returned for Query {i+1}. Skipping.")
        
    # 3. Create and save the final DataFrame
    if not results:
        print("\nFATAL: No recommendations were generated. Cannot create submission file.")
        return

    # Ensure the columns match the required order: Query, Assessment_url
    df_submission = pd.DataFrame(results, columns=['Query', 'Assessment_url']) 
    df_submission.to_csv(output_path, index=False)
    
    print("\n--- Submission Generation Complete ---")
    print(f"Total entries generated: {len(df_submission)}")
    print(f"Submission file saved to: {output_path}")

    # 4. Output the result as a Markdown Table (preview)
    # We'll display only the first 20 rows for a cleaner console preview.
    print("\n--- Submission File Preview (First 20 Rows) ---")
    df_preview = df_submission.head(20).copy()
    
    # Optional: Truncate the Query text for better table readability in the console
    df_preview['Query'] = df_preview['Query'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    print(df_preview.to_markdown(index=False))
    print("-------------------------------------------------")


def main():
    """Main execution function."""
    
    # 1. Health Check
    try:
        health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if health_response.status_code != 200:
            raise Exception(f"Health Check Failed: HTTP {health_response.status_code}")
        health_data = health_response.json()
        if health_data.get('status') != 'ok':
            raise Exception(f"Health Check Status Not OK: {health_data}")
        print(f"API Health Check Passed. Vector store loaded: {health_data.get('vector_store_loaded')}")
        
    except requests.exceptions.RequestException as e:
        print("\n" + "="*50)
        print("ERROR: FastAPI Server Not Running or Inaccessible!")
        print(f"Please ensure you run 'uvicorn api:app --reload' in your terminal.")
        print(f"Attempted connection to: {API_BASE_URL}. Error: {e}")
        print("="*50 + "\n")
        return

    # 2. Load Queries
    test_queries = load_test_queries(TEST_FILE_NAME)
    if test_queries is None:
        print("Could not proceed without test queries.")
        return

    # 3. Generate Submission
    generate_submission(test_queries, OUTPUT_FILE_NAME)

if __name__ == '__main__':
    main()