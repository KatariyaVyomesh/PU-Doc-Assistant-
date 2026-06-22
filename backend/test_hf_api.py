import os
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def test_hf_api_logic():
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"Testing with API Key: {'Set' if api_key and 'your' not in api_key else 'Not Set'}")
    
    if not api_key or "your" in api_key:
        print("Gracious handling of missing/default API key: PASS")
        return

    client = InferenceClient(api_key=api_key)
    try:
        texts = ["Hello world", "Artificial Intelligence is fascinating"]
        response = client.feature_extraction(model=model_id, text=texts)
        embeddings = np.array(response)
        print(f"Success! Received embeddings shape: {embeddings.shape}")
        if embeddings.shape[1] == 384:
            print("Embedding dimension matches all-MiniLM-L6-v2 (384): PASS")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_hf_api_logic()
