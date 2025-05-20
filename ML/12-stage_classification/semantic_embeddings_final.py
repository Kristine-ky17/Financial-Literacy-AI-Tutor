import os
import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

def get_embedding(text):
    """
    Generate an embedding for the provided text using the text-embedding-ada-002 model.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def main():

    filepath = "synthetic_data_all_stages.csv"
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    df['generated_response'] = df['generated_response'].astype(str).fillna("")

    print("Generating embeddings for the generated_response column...")
    df['embeddings'] = df['generated_response'].apply(get_embedding)

    output_file = "processed_all_stages_embeddings.csv"
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
