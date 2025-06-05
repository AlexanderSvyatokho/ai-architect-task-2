import re
import chromadb
from openai import OpenAI
from tqdm import tqdm

OLLAMA_API = "http://localhost:11434/v1"
MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text" 
RAG_RESULTS = 3 # number of results to return from RAG

ollama_client = OpenAI(base_url=OLLAMA_API, api_key='ollama')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("nodejs_best_practices")

def prepare_rag_index():
    print("Preparing RAG index...")

    # 1. Read the file
    text = read_file()

    # 2. Split by sections based on headers
    chunks = split_by_headers(text)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk['header']}")
    #     print(f"Content: {chunk['content'][:100]}...")

    # 3. Create embeddings for each chunk
    texts = [chunk['text'] for chunk in chunks]
    embeddings = create_embeddings(texts)
    # for i, embedding in enumerate(embeddings):
    #     print(f"Embedding {i+1}: {embedding[:10]}...")

    # 4. Store in vector database
    store_in_vector_db(chunks, embeddings)

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

def create_embeddings(texts):
    print("Creating embeddings...")
    embeddings = []
    
    for i, text in tqdm(enumerate(texts)):
        # Create embedding using Ollama API
        response = ollama_client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

def store_in_vector_db(chunks, embeddings):
    print("Storing in vector database...")
       
    # Prepare data for Chroma
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [{'header': chunk['header']} for chunk in chunks]
    
    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Stored {len(chunks)} chunks in vector database")

def query_rag(collection, question):
    # Create embedding for the question
    question_response = ollama_client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    )
    question_embedding = question_response.data[0].embedding
    
    # Search similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=RAG_RESULTS
    )
    
    # Get relevant context
    context = "\n\n".join(results['documents'][0])

    # Create prompt with context
    prompt = f"""Based on the following Node.js best practices documentation, answer the question. If the answer is not in the documentation, say "Sorry, I can't answer based on the available documentation".

Context:
{context}

Question: {question}

Answer:"""
    
    # Get response from LLM
    response = ollama_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides answers about Node.js best practices based on the provided documentation."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def test_rag():
    print("\n\n*** Testing RAG system with sample questions... ***")

    questions = [
        "How should I document API errors?",
        "Is it okay to use var in Node.js?",
        "How should I organize my tests?"
    ]
    
    for question in questions:
        answer = query_rag(collection, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print("-" * 50)

    print("*** Testing complete. You can now interact with the RAG system. ***\n\n")

def chat_loop():
    print("Welcome to the Node.js Best Practices RAG system!")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        answer = query_rag(collection, question)
        print(f"\nAnswer: {answer}")
        print("-" * 50)

prepare_rag_index()
test_rag()
chat_loop()
