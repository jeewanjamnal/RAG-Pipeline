# PDF-Based RAG Application

A basic Retrieval-Augmented Generation (RAG) application built using Node.js, Express.js, and Google Gemini.

## Tech Stack

- Node.js & Express.js
- Multer – PDF file upload
- pdf-parse – PDF text extraction
- Gemini API – Embeddings and answer generation
- Cosine Similarity – Finding relevant document chunks

## Installation

```bash
npm init -y
npm install express multer pdf-parse @google/genai dotenv
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

## How It Works

### 1. PDF Upload

```javascript
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('pdf'), async (req, res) => {
```

Multer handles a single PDF upload and temporarily stores it in the `uploads/` directory.

### 2. Extract PDF Text

```javascript
const bufferData = fs.readFileSync(req.file.path);
const pdfData = await pdfparse(bufferData);
const text = pdfData.text;
```

Extracts readable text from the uploaded PDF.

### 3. Split Text into Chunks

```javascript
const chunks = text
    .split('\n\n')
    .filter(chunk => chunk.trim() !== '');
```

Splits extracted text into smaller sections for embedding and retrieval.

### 4. Generate Embeddings

```javascript
async function createEmbedding(text) {
    const response = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: text,
    });

    return response.embeddings[0].values;
}
```

Converts text into a numerical vector (embedding) that represents its semantic information.

Generate an embedding for each document chunk and store it with the original text:

```javascript
chunkEmbeddings.push({
    chunk,
    embedding
});
```

### 5. Generate Question Embedding

```javascript
const question = req.body.question;
const questionEmbedding = await createEmbedding(question);
```

Converts the user's question into an embedding so it can be compared with document chunk embeddings.

### 6. Calculate Cosine Similarity

```javascript
function cosineSimilarity(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
        throw new Error("Vectors must have the same dimensions");
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
        magnitudeA += vectorA[i] ** 2;
        magnitudeB += vectorB[i] ** 2;
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) return 0;

    return dotProduct / (magnitudeA * magnitudeB);
}
```

Measures similarity between two vectors by calculating the cosine of the angle between them.

**Formula:**

Cosine Similarity =  A.B / |A|*|B|

### 7. Retrieve Best Matching Chunk

```javascript
let bestChunk = null;
let bestScore = -Infinity;

for (const { chunk, embedding } of chunkEmbeddings) {
    const score = cosineSimilarity(questionEmbedding, embedding);

    if (score > bestScore) {
        bestScore = score;
        bestChunk = chunk;
    }
}
```

Compares the question embedding with every chunk embedding and retrieves the chunk with the highest similarity score.

### 8. Generate Answer Using Gemini

```javascript
const result = await ai.models.generateContent({
    model: 'gemini-3.5-flash-lite',
    contents: 'Answer the question ' + question +
        ' using this context: ' + bestChunk,
});
```

Sends the user's question and retrieved document context to Gemini to generate an answer.

### 9. Return Response

```javascript
return res.status(200).json({
    matchedChunk: bestChunk,
    matchedScore: bestScore,
    response: result.text
});
```

Returns the retrieved chunk, its cosine similarity score, and the generated answer.

## API Usage

**Endpoint:** `POST /upload`

**Content-Type:** `multipart/form-data`

| Key | Type | Value |
|---|---|---|
| pdf | File | Upload PDF |
| question | Text | Ask a question about the PDF |

**Run the application:**

```bash
node server.js
```

Server URL: `http://localhost:3000`

## RAG Workflow

```text
PDF Upload
    ↓
Extract Text
    ↓
Split into Chunks
    ↓
Generate Chunk Embeddings
    ↓
Generate Question Embedding
    ↓
Calculate Cosine Similarity
    ↓
Retrieve Best Matching Chunk
    ↓
Send Context + Question to Gemini
    ↓
Generate Answer
    ↓
Return JSON Response
```

## Key Learnings

- PDF upload and text extraction
- Text chunking
- Vector embeddings
- Cosine similarity
- Semantic search and retrieval
- Retrieval-Augmented Generation (RAG)