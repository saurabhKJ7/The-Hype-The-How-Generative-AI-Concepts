# Indian Legal Document Search System

A sophisticated legal document search engine that supports multiple similarity techniques for searching through Indian legal documents.

## Features

- Upload and process legal documents (PDF/Word)
- Support for various Indian legal document types:
  - Indian Income Tax Act
  - GST Act provisions
  - Court judgments
  - Property law documents
- Multi-method similarity search with side-by-side comparison
- Real-time evaluation metrics
- Legal entity extraction
- Interactive visualization of search results

## Similarity Methods

1. Cosine Similarity
2. Euclidean Distance
3. Maximal Marginal Relevance (MMR)
4. Hybrid Similarity (Cosine + Legal Entity Match)

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_lg
   ```
5. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Project Structure

```
.
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── frontend/
├── data/
├── tests/
└── requirements.txt
```

## Evaluation Metrics

- Precision@5
- Recall
- Diversity Score (for MMR)

## License

MIT License 