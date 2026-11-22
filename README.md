# Sam's JobMatch AI 🎯

An AI-powered semantic job search assistant that helps users discover relevant career opportunities using natural-language queries.

## Features

- Natural-language job search
- Semantic retrieval using Sentence Transformers
- Vector similarity ranking using cosine similarity
- SQLite database for job listings
- LLM-powered recommendations and explanations
- Function/tool calling with Groq
- Interactive Streamlit interface

## Architecture

User Query → Embedding Model → Semantic Search → SQLite Job Database → Retrieved Results → LLM Tool Calling → AI Recommendations

## Tech Stack

- Python
- Streamlit
- SQLite
- Sentence Transformers (all-MiniLM-L6-v2)
- NumPy
- Groq API

## How it works

1. Job listings are stored in SQLite.
2. Job descriptions are converted into vector embeddings.
3. User queries are embedded using the same model.
4. Cosine similarity ranks relevant jobs.
5. The LLM invokes the search tool to retrieve matching jobs.
6. The LLM generates recommendations with explanations.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY="your_api_key"
```

Run the application:

```bash
streamlit run app.py
```

## Disclaimer

The application currently uses sample job listings for demonstration purposes. Similarity scores indicate semantic relevance, not actual hiring probability.
