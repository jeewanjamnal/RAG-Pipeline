const express = require('express');
const multer = require('multer');
const pdfparse = require('pdf-parse');
const fs = require('fs');
const {GoogleGenAI} = require('@google/genai');
require('dotenv').config();

const app = express();
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

const upload = multer({ dest: 'uploads/'});

async function createEmbedding(text) {
    const response = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: text,
    });

    return response.embeddings[0].values;
}


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

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
}


app.post('/upload', upload.single('pdf'),async(req, res) => {
    if(!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    // console.log(req.file);

    try {
        const bufferData = fs.readFileSync(req.file.path);
        const pdfData = await pdfparse(bufferData);
        const text = pdfData.text;

        const chunks = text.split('\n\n').filter(chunk => chunk.trim() !== ''); // Split by double newlines and filter out empty chunks
       
        const chunkEmbeddings = [];
        for (const chunk of chunks) {
            const embedding = await createEmbedding(chunk);
            chunkEmbeddings.push({
                chunk: chunk,
                embedding
            });
        }
        
        const question = req.body.question;
        const questionEmbedding = await createEmbedding(question);
        let bestChunk = null;
        let bestScore = -Infinity;
         for (const {chunk, embedding} of chunkEmbeddings) {
            const score = cosineSimilarity(questionEmbedding, embedding);
            if (score > bestScore) {
                bestScore = score;
                bestChunk = chunk;
            }
        }

        if(!bestChunk) {
            return res.status(404).json({
                success: false,
                message: 'No matching chunk found for the question',
                data: null
            });
        }
        const result = await ai.models.generateContent({
            model: 'gemini-3.5-flash-lite',
            contents: 'Answer the question ' + question + ' using this context: ' + bestChunk,
        });
        return res.status(200).json({
            matchedChunk: bestChunk,
            matchedScore: bestScore,
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