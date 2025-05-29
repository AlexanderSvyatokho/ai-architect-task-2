print("Starting the Ollama RAG example...")

from openai import OpenAI

OLLAMA_API = "http://localhost:11434/v1"
MODEL = "llama3.2"

messages = [
    {"role": "user", "content": "Describe some of the business applications of Generative AI"}
]

ollama_via_openai = OpenAI(base_url=OLLAMA_API, api_key='ollama')

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)