# Baby RAG System 🚀

A simple Retrieval-Augmented Generation (RAG) system that demonstrates how to:
- Convert text snippets into vector embeddings
- Store them in a vector database (Chroma)
- Retrieve relevant information using natural language queries

## What This Does

This is a "baby" RAG system that takes 15 technology-related text snippets, converts them into embeddings, stores them in a vector database, and allows you to ask natural language questions to find the most relevant information.

## Features

- ✅ 15 curated text snippets about technology topics
- ✅ Vector embeddings using sentence-transformers
- ✅ Local vector storage with ChromaDB
- ✅ Natural language query interface
- ✅ Similarity scoring and ranking
- ✅ Interactive demo mode

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
2. Generate embeddings
3. Store them in ChromaDB
4. Run through demo queries
5. Enter interactive mode for your own questions

### Example Queries

Try asking questions like:
- "What is machine learning?"
- "How do neural networks work?"
- "Tell me about Python programming"
- "What are databases?"
- "Explain cloud computing"

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

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Vector Database**: ChromaDB (in-memory)
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
- 2GB RAM (for embedding model)
- Internet connection (for initial model download)

Enjoy exploring your baby RAG system! 🎉
