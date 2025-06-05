# RAG sample app: Node.js best practices

This is a simple demonstration of the RAG system.

How it works:
1. The input data for RAG is an MD file with Node.js best practices taken from https://github.com/goldbergyoni/nodebestpractices. The copy of the MD file is stored in `data/node-best-practices.md`
1. The input file is split by sections, embeddings are created and stored in-memory
1. The app starts with running 3 sample questions through RAG
1. After the sample questions are answered automatically a chat loop starts, where a user can enter questions in the terminal

Technical characterists:
1. Lightweight llama3.2 3B LLM is used for output generation
1. nomic-embed-text embedding model used to create embeddings
1. Both models are run "locally" using Ollama
1. Embeddings are stored locally using vector DB Chroma

Known limitations:
1. Since chunks are created per section, the best results are generated for questions that target specific problems/best practices. For example, asking "List most important best practices" will not yield good results as this would require summarizing the entire document. 


## Run Instructions
You can run the app using Docker or by preparing the local environment.

### Build and run in Docker
```
docker build -t rag-app .
docker run -it rag-app
```

### Local Execution

Prerequisites: 
- python3 installed
- Ollama (https://ollama.com/) installed and running locally at http://localhost:11434 

Pull Ollama's llama3.2 and nomic-embed-text LLMs:
```
ollama pull llama3.2
ollama pull nomic-embed-text
```

Install dependencies:
``` 
pip3 install -r requirements.txt

```

Run the application:
```
python3 src/rag.py
```