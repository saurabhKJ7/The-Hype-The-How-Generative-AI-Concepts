# Intelligent Document Chunker

An enterprise-grade document processing pipeline that intelligently chunks and processes documents based on their content type for improved retrieval accuracy.

## Features

- Document type classification (Technical docs, API references, Support tickets, Policies, Tutorials)
- Adaptive chunking strategies based on document type
- Vector embeddings generation and storage
- Modular and extensible architecture
- Performance monitoring and optimization

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
.
├── app/
│   ├── api/              # FastAPI routes
│   ├── core/             # Core application logic
│   ├── models/           # Pydantic models
│   └── services/         # Business logic services
├── tests/                # Test files
├── .env.example          # Example environment variables
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Usage

1. Document Processing:
   - Upload documents through the API endpoint
   - Documents are automatically classified and chunked
   - Chunks are embedded and stored in the vector database

2. Retrieval:
   - Query the processed documents through the API
   - Get relevant chunks based on semantic similarity

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 