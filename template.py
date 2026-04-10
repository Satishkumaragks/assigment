from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import openai

load_dotenv()

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = os.getenv("API_TOKEN2")

HEADERS = {
    "useLegacyCompletionsEndpoint": "false",    
    "X-Tenant-ID": "default_tenant"
}

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers=HEADERS
)

def list_models():
    models = client.models.list()
    # // Print the list of models
    for i, model in enumerate(models):
        print(f"{i + 1}. {model.id}")



def get_models(model=str, temperature=float, max_tokens=int):
    return ChatOpenAI(
        model=model,
        api_key=API_KEY,
        base_url=BASE_URL,
        default_headers=HEADERS,
        temperature=temperature,
        max_tokens=max_tokens
    )
