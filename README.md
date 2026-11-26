# PDF RAG using Node.js, Gemini & Qdrant

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, store their embeddings in Qdrant, and ask questions based on the document content.

## Tech Stack

- **Node.js & Express** – Backend API
- **Multer** – PDF file uploads
- **pdf-parse** – Extract text from PDFs
- **Google Gemini** – Generate embeddings and answers
- **Qdrant** – Vector database for similarity search
- **UUID** – Generate unique IDs for document chunks

## Installation

Install the required dependencies:

```bash
npm install express multer pdf-parse @qdrant/js-client-rest @google/genai dotenv
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## How It Works

1. Upload a PDF using Multer.
2. Extract text from the PDF using `pdf-parse`.
3. Split extracted text into smaller chunks.
4. Generate embeddings for each chunk using Gemini.
5. Store embeddings and text payloads in Qdrant.
6. Convert the user's question into an embedding.
7. Search Qdrant for the most similar chunks using cosine similarity.
8. Send the retrieved context and question to Gemini.
9. Return the generated answer.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/create-collection` | Create Qdrant collection |
| POST | `/upload` | Upload PDF and ask a question |

### Upload API

For `/upload`, use `multipart/form-data` with:

- `pdf` – PDF file
- `question` – Question about the PDF

## Key Learning Concepts

- Text extraction and chunking
- Vector embeddings
- Cosine similarity
- Vector databases and Qdrant
- Semantic search
- Retrieval-Augmented Generation (RAG)

## Understanding Cosine Similarity

Cosine similarity measures how similar two vectors are based on the angle between them.

### Cosine Similarity Score Interpretation

| Cosine Score | Interpretation |
|---|---|
| Close to 1.0 | Vectors point in nearly the same direction |
| 0.8 – 0.99 | Often indicates strong similarity |
| 0.5 – 0.79 | Potentially related |
| 0 – 0.49 | Weak directional similarity |
| Close to 0 | Nearly perpendicular vectors |
| Negative | Vectors point in opposite directions |

### Example in RAG

Suppose the user asks:

**"What is machine learning?"**

Qdrant compares the question embedding with stored document embeddings and ranks the chunks based on cosine similarity.

| Retrieved Chunk | Cosine Score | Relevance |
|---|---:|---|
| Machine learning is a subset of AI. | 0.95 | High |
| AI allows computers to learn. | 0.87 | High |
| Python is a programming language. | 0.42 | Low |
| The weather is sunny today. | 0.08 | Very Low |

### Interpretation

- Higher cosine scores generally indicate more relevant chunks.
- Lower scores generally indicate less directional similarity.
- Qdrant ranks matching vectors based on their similarity scores.
- The highest-scoring chunks are retrieved and passed to Gemini as context.

**Note:** These score ranges are illustrative, not universal relevance thresholds. Actual relevance depends on your embedding model and data. A high cosine score does not guarantee that a chunk answers the question.

## Future Improvements

- Separate PDF upload and question-answering APIs.
- Retrieve multiple relevant chunks instead of just one.