const express = require('express');
const multer = require('multer');
const pdfparse = require('pdf-parse');
const fs = require('fs');
const { QdrantClient } = require('@qdrant/js-client-rest');
const { GoogleGenAI } = require('@google/genai');
const crypto = require('crypto');
require('dotenv').config();

const app = express();
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});
const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL, 
    apiKey: process.env.QDRANT_API_KEY
});

const upload = multer({ dest: 'uploads/'});

async function createEmbedding(text) {
    const response = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: text,
    });

    return response.embeddings[0].values;
}

// Generates a deterministic UUID-shaped ID from chunk text using SHA-256 hashing.
// Ensures identical chunks always generate the same ID to prevent duplicate points in Qdrant.
function generateChunkId(chunk) {
    return crypto
        .createHash('sha256')
        .update(chunk.trim())
        .digest('hex')
        .slice(0, 32)
        .replace(
            /^(.{8})(.{4})(.{4})(.{4})(.{12})$/,
            '$1-$2-$3-$4-$5'
        );
}

app.get('/create-collection', async (req, res) => {
   try{
    await qdrant.createCollection('pdf-collection',{
        vectors:{
            size: 3072,
            distance: 'Cosine'
        }
    });
    res.status(200).json({ message: 'Collection created successfully' });
   }catch(error){
    res.status(500).json({ error: 'Failed to create collection' });
   }
});


app.post('/upload', upload.single('pdf'),async(req, res) => {
    if(!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    try {
        const bufferData = fs.readFileSync(req.file.path);
        const pdfData = await pdfparse(bufferData);
        const text = pdfData.text;

        const chunks = text.split('\n\n').filter(chunk => chunk.trim() !== ''); 
       
        const chunkEmbeddings = [];
        for (const chunk of chunks) {
            const embedding = await createEmbedding(chunk);
            chunkEmbeddings.push({
                chunk: chunk,
                embedding
            });
        }

        const points = chunkEmbeddings.map(({chunk, embedding}) => ({
            id: generateChunkId(chunk),
            vector: embedding,
            payload: { text: chunk }
        }));

        await qdrant.upsert('pdf-collection', {
            points
        });
        
        const question = req.body.question;
        const questionEmbedding = await createEmbedding(question);

        const searchResult = await qdrant.query('pdf-collection', {
            query: questionEmbedding,
            limit: 5, // return the top 5 most relevant chunks
            with_payload: true
        });

        if (searchResult.points.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'No matching chunk found',
                data: null
            });
        }

        // Based on limit 1 get one best chunk and score from searchResult, but now we are using 5 relevant matched chunks for context.

        // const bestChunk = searchResult.points[0].payload.text;
        // const bestScore = searchResult.points[0].score;

        // const result = await ai.models.generateContent({
        //     model: 'gemini-3.5-flash-lite',
        //     contents: 'Answer the question ' + question + ' using this context: ' + bestChunk,
        // });

        const context = searchResult.points
        .map((point, index) => {
            return `Chunk ${index + 1}:\n${point.payload.text}`;
        })
        .join('\n\n');


        const result = await ai.models.generateContent({
            model: 'gemini-3.5-flash-lite',
            contents: `
                You are a helpful, knowledgeable, and conversational AI assistant.
                Your task is to answer the user's question naturally, just like a human expert would in a real conversation.

                USER QUESTION:
                ${question}

                REFERENCE CONTEXT:
                ${context}

                GUIDELINES:

                1. Understand the user's question carefully before answering.
                2. Use the provided context as your primary source of information.
                3. Read and combine information from multiple relevant chunks to provide a complete and meaningful answer.
                4. Write naturally and conversationally. Avoid robotic, overly formal, or textbook-style language.
                5. Start directly with the answer instead of using unnecessary introductions like "Based on the provided context" or "According to the document".
                6. Explain concepts clearly using simple language that is easy to understand.
                7. When appropriate, organize the answer using short paragraphs, bullet points, numbered steps, or examples.
                8. Avoid repeating the same information or adding unnecessary details.
                9. Do not mention chunks, embeddings, vector databases, retrieval, or internal processing.
                10. Never invent facts or use outside knowledge to fill gaps in the provided context.
                11. If the context does not contain enough information to answer, clearly and naturally tell the user what information is missing.
                12. If only part of the question can be answered, answer that part and explain what cannot be determined.

                RESPONSE STYLE:
                - Friendly, natural, and professional.
                - Clear and concise, but sufficiently detailed.
                - Human-like conversational flow.
                - No unnecessary greetings or closing statements.
                - Use Markdown formatting only when it improves readability.

                Now answer the user's question.
            `,
        });


        
        return res.status(200).json({
            matchedChunks: context,
            response: result.text
        });
    }catch (error) {
        console.error('Error parsing PDF:', error);
        res.status(500).json({ error: error });
    }finally {
        fs.unlinkSync(req.file.path);
    }
});


app.listen(3000, () => {
  console.log(`Server is running......`);
});