# SHL Assessment Recommendation System (RAG)

This project implements a Retrieval-Augmented Generation (RAG) system designed to recommend relevant SHL talent assessments based on a natural language job description query. The system leverages vector embeddings for semantic search against a simulated catalog of assessments and uses a FastAPI service to expose the recommendation logic via an API.

# Project Structure

The repository is structured to separate the data pipeline, model generation, API service, and testing utility.
├── Gen_AI_Dataset_Train.csv      # Training data (simulated catalog source)
├── Gen_AI Dataset.xlsx - Test-Set.csv # Queries used for submission generation
├── data_pipeline.py              # Loads and structures the catalog data.
├── embedding_generator.py        # Generates vector embeddings and creates the Vector Store.
├── api.py                        # FastAPI service exposing the `/recommend` endpoint (the RAG engine).
├── submission_generator_rag.py   # Utility script to test the API with a batch of queries or a single interactive query, and generate the final submission file.
└── model_assets_rag/             # Directory to store generated model assets (created upon running scripts)
    ├── catalog_data.pkl          # Structured assessment data (output of data_pipeline.py)
    └── assessment_vector_store.pkl # The final vector store (embeddings + metadata) (output of embedding_generator.py)
    
# Setup and Installation

Follow these steps to set up the project environment.

# 1. Prerequisites

Ensure you have Python (3.8+) installed.

# 2. Install Dependencies

All necessary Python packages are listed in requirements.py (which includes pandas, requests, fastapi, uvicorn, scikit-learn, numpy, and joblib).
# Install required libraries
pip install pandas requests fastapi uvicorn pydantic joblib scikit-learn numpy

# IMPORTANT: The submission generator requires the 'tabulate' library for console output
pip install tabulate
Usage

The project requires a three-step process: (1) Prepare Data, (2) Start the API Service, and (3) Run the Generator Script.

Step 1: Prepare the Data and Vector Store

You must run the data pipeline and embedding generator once to create the necessary model assets in the model_assets_rag directory.
# 1. Structure the catalog data
python data_pipeline.py

# 2. Generate vector embeddings and save the vector store
python embedding_generator.py

Step 2: Start the Recommendation API Service

The FastAPI server (api.py) must be running in the background to handle recommendation requests.

Open Terminal 1 and run:
uvicorn api:app --reload

The service will be accessible at http://127.0.0.1:8000. You should see an output indicating the vector store was loaded successfully.

Step 3: Run the Submission Generator Utility

The submission_generator_rag.py script now supports two modes:

Full Submission (file mode): Runs all queries from the test file and generates submission_rag.csv.

Interactive Test (test mode): Allows you to input a single, custom job description query for quick testing.

Open Terminal 2 and run:
python submission_generator_rag.py

When prompted:

Type file to run the full batch test.

Type test and then paste your custom query to test the RAG engine interactively.

Example Interactive Test Session:
$ python submission_generator_rag.py
API Health Check Passed. Vector store loaded: True
Enter mode ('file' for full submission or 'test' for single query): test
Enter your job description query: We need to hire a project manager with strong communication skills and experience leading agile development teams.
Running single query test. Output will be saved to: single_query_test_output.csv

--- Starting Submission Generation (1 Queries) ---
Processing Query 1/1: 'We need to hire a project manager with strong communication skills and experie...'
... (API calls happen here)

--- Submission Generation Complete ---
Total entries generated: 10
Submission file saved to: single_query_test_output.csv

--- Submission File Preview (First 20 Rows) ---
| Query | Assessment_url |
|:---|:---|
| We need to hire a project ma... | [https://www.shl.com/solutions/products/product-catalog/view/agile-leadership-assessment/](https://www.shl.com/solutions/products/product-catalog/view/agile-leadership-assessment/) |
| We need to hire a project ma... | [https://www.shl.com/products/product-catalog/view/professional-7-1/](https://www.shl.com/products/product-catalog/view/professional-7-1/) |
| ... | ... |
-------------------------------------------------

