print("Starting the Ollama RAG example...")

import re
from openai import OpenAI

OLLAMA_API = "http://localhost:11434/v1"
MODEL = "llama3.2"

def prepare_rag_index():
    print("Preparing RAG index...")
    text = read_file()
    chunks = split_by_headers(text)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk['header']}")
    #     print(f"Content: {chunk['content'][:10000]}...")

def read_file():
    file_path = "data/node-best-practices.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print("Loaded content from file.")
    except Exception as e:
        print(f"Error loading file: {e}")
        text = ""
    return text

def split_by_headers(text):
    print("Splitting document by headers...")
    
    # Split by headers (# or ##)
    sections = re.split(r'\n(#{1,2}\s+.+)', text)
    
    chunks = []
    current_header = ""
    current_content = ""
    
    for i, section in enumerate(sections):
        if section.startswith('#'):
            # Save previous chunk if it has content
            if current_header and current_content.strip():
                chunks.append({
                    'header': current_header.strip(),
                    'content': current_content.strip(),
                    'text': f"{current_header}\n{current_content}".strip()
                })
            
            # Start new chunk
            current_header = section
            current_content = ""
        else:
            current_content += section
    
    # Don't forget the last chunk
    if current_header and current_content.strip():
        chunks.append({
            'header': current_header.strip(),
            'content': current_content.strip(),
            'text': f"{current_header}\n{current_content}".strip()
        })
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def test_prompt():
    messages = [
        {"role": "user", "content": "Describe some of the business applications of Generative AI"}
    ]

    ollama_via_openai = OpenAI(base_url=OLLAMA_API, api_key='ollama')

    response = ollama_via_openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()  # for a newline after streaming is done

# test_prompt()

prepare_rag_index()
