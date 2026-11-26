const express = require('express');
const multer = require('multer');
const pdfparse = require('pdf-parse');
const fs = require('fs');
const { QdrantClient } = require('@qdrant/js-client-rest');
const { GoogleGenAI } = require('@google/genai');
const { v4: uuidv4 } = require('uuid');
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
            id: uuidv4(),
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
            limit: 1,
            with_payload: true
        });

        if (searchResult.points.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'No matching chunk found',
                data: null
            });
        }

        const bestChunk = searchResult.points[0].payload.text;
        const bestScore = searchResult.points[0].score;

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