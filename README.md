# Baby RAG System 🚀

A simple Retrieval-Augmented Generation (RAG) system that demonstrates how to:
- Convert text snippets into vector embeddings
- Store them in a vector database (Chroma)
- Retrieve relevant information using natural language queries

## What This Does

This is a "baby" RAG system that takes 15 technology-related text snippets, converts them into embeddings, stores them in a vector database, and allows you to ask natural language questions to find the most relevant information.

## Features

- ✅ 15 curated text snippets about technology topics
- ✅ Lightweight TF-IDF vector search (scikit-learn)
- ✅ No heavy model downloads required
- ✅ Natural language query interface
- ✅ Similarity scoring and ranking
- ✅ Interactive demo mode
- ✅ FastAPI endpoint for programmatic access

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Demo

```bash
python rag_system.py
```

This will:
1. Load the text snippets
2. Build a TF-IDF vector index
3. Run through demo queries
4. Enter interactive mode for your own questions

### Example Queries

Try asking questions like:
- "What is machine learning?"
- "How do neural networks work?"
- "Tell me about Python programming"
- "What are databases?"
- "Explain cloud computing"

### Run the API Server

```bash
uvicorn rag_api:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

### API Endpoints

- `GET /health` — verify the service is running
- `POST /query` — ask questions (`{"question": "...", "n_results": 3}`)
- `GET /collection/info` — metadata about the stored snippets
- `GET /collection/snippets` — retrieve the full dataset

### Run the MCP Server

```bash
python rag_mcp_server.py
```

Add this server to your Cursor `mcpServers` configuration, e.g.:

```json
{
  "name": "Baby RAG",
  "command": "python",
  "args": ["rag_mcp_server.py"]
}
```

The MCP server exposes a single tool `rag_query` that accepts:

- `question` (string, required)
- `n_results` (integer, optional, default 3, max 10)

It returns the same payload as the FastAPI `/query` endpoint.

Visit `http://localhost:3333/playground` (after starting the MCP server) for an HTML harness that walks through session initialization and lets you send custom JSON-RPC payloads.

## How It Works

1. **Data Loading**: Loads 15 text snippets from `data/snippets.json`
2. **Embedding Generation**: Uses `all-MiniLM-L6-v2` model to create vector representations
3. **Vector Storage**: Stores embeddings in ChromaDB with metadata
4. **Query Processing**: Converts your question into an embedding
5. **Similarity Search**: Finds the most similar text snippets using cosine similarity
6. **Results Ranking**: Returns results sorted by similarity score

## File Structure

```
├── rag_system.py          # Main RAG system implementation
├── data/
│   └── snippets.json      # Text snippets dataset
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technical Details

- **Vector Index**: TF-IDF (scikit-learn)
- **Similarity Metric**: Cosine similarity
- **Language**: Python 3.7+

## Demo Results

The system can successfully answer questions like:
- "What is machine learning?" → Returns the machine learning snippet
- "How do neural networks work?" → Returns the neural networks snippet
- "Tell me about Python" → Returns the Python basics snippet

## Next Steps

This is a foundation that can be extended with:
- Larger datasets
- More sophisticated embedding models
- Persistent vector storage
- Integration with LLMs for generation
- Web interface
- Advanced filtering and metadata search

## Requirements

- Python 3.7+
- ~1GB RAM (lightweight TF-IDF index)
- No external model download required

Enjoy exploring your baby RAG system! 🎉
