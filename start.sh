#!/bin/bash
set -e

ollama serve &
sleep 5

ollama pull llama3.2
ollama pull nomic-embed-text
python3 src/rag.py
