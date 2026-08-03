const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const Busboy = require('busboy');
const sharp = require('sharp');

const app = express();

// --- Configurações ---
const PORT = process.env.PORT || 3000;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const GEN_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b';
const EDIT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const NVIDIA_TIMEOUT = 120000; // 2 minutos
const MAX_RETRIES = 2;

if (!NVIDIA_API_KEY) {
  console.error('[FATAL] NVIDIA_API_KEY não definida.');
}

app.use(cors());
app.use(express.json());

// --- Utilitários ---
function parseFormData(req) {
  return new Promise((resolve, reject) => {
    const busboy = Busboy({ 
      headers: req.headers, 
      limits: { fileSize: MAX_FILE_SIZE, files: 1 },
      defCharset: 'utf8'
    });
    const fields = {};
    const files = {};
    let fileCount = 0;

    busboy.on('file', (fieldname, stream, info) => {
      fileCount++;
      // Validação estrita de tipo e quantidade
      if (fileCount > 1 || !ALLOWED_MIME_TYPES.includes(info.mimeType)) {
        stream.resume(); // Descarta o stream
        return reject(new Error('Arquivo inválido ou múltiplos uploads não permitidos.'));
      }

      const chunks = [];
      stream.on('data', (chunk) => chunks.push(chunk));
      stream.on('end', () => {
        files[fieldname] = {
          buffer: Buffer.concat(chunks),
          filename: info.filename,
          mimetype: info.mimeType,
        };
      });
      stream.on('error', reject);
    });

    busboy.on('field', (fieldname, val) => { fields[fieldname] = val; });
    busboy.on('close', () => resolve({ fields, files }));
    busboy.on('error', reject);
    
    req.pipe(busboy);
  });
}

async function requestNvidia(url, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), NVIDIA_TIMEOUT);

  try {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': `Bearer ${NVIDIA_API_KEY}`,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        // Retorna imediatamente se sucesso ou erro de cliente (não tentar de novo em 400/422/429)
        if (response.status < 500 || attempt === MAX_RETRIES) {
          return response;
        }

        console.warn(`[NVIDIA RETRY] Status ${response.status}, tentativa ${attempt}/${MAX_RETRIES}`);
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      } catch (err) {
        if (err.name === 'AbortError') throw new Error('Timeout na requisição à NVIDIA.');
        if (attempt === MAX_RETRIES) throw err;
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      }
    }
  } finally {
    clearTimeout(timeout);
  }
}

// --- Rotas ---
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Bunix Image API',
    endpoints: { generate: '/generate' }
  });
});

app.post('/generate', async (req, res) => {
  try {
    const { fields, files } = await parseFormData(req);
    const prompt = (fields.param || fields.prompt || '').trim();

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt obrigatório. Forneça uma descrição detalhada.' });
    }

    const hasImage = files.image && files.image.buffer && files.image.buffer.length > 0;
    let nvidiaResponse;

    if (hasImage) {
      const dataUri = `data:${files.image.mimetype};base64,${files.image.buffer.toString('base64')}`;
      nvidiaResponse = await requestNvidia(EDIT_URL, {
        prompt, image: dataUri, samples: 1, seed: 0, steps: 30,
      });
    } else {
      nvidiaResponse = await requestNvidia(GEN_URL, {
        prompt, samples: 1, seed: 0, steps: 4,
      });
    }

    const rawText = await nvidiaResponse.text();
    let data;
    
    try {
      data = JSON.parse(rawText);
    } catch {
      console.error('[NVIDIA PARSE ERROR]', nvidiaResponse.status, rawText.slice(0, 500));
      return res.status(500).json({ error: 'Resposta inválida do provedor de IA.' });
    }

    if (!nvidiaResponse.ok) {
      if (nvidiaResponse.status === 429) {
        return res.status(429).json({ error: 'Limite de taxa excedido. Aguarde e tente novamente.' });
      }
      return res.status(nvidiaResponse.status || 500).json({
        error: 'Falha ao gerar a imagem.',
        nvidia_details: data,
      });
    }

    // Extração da imagem (Compatibilidade NIM vs OpenAI)
    const artifact = data?.artifacts?.[0];
    const base64Image = artifact?.base64 || data?.data?.[0]?.b64_json;

    if (artifact?.finishReason === 'CONTENT_FILTERED' || !base64Image) {
      return res.status(422).json({
        error: 'Conteúdo bloqueado pelas políticas de segurança da NVIDIA.',
        nvidia_details: data,
      });
    }

    // OTIMIZAÇÃO PURA: Apenas conversão e compressão sem filtros visuais
    // Isso garante que a imagem da IA seja entregue sem deformações artificiais
    const rawBuffer = Buffer.from(base64Image, 'base64');
    let finalBuffer = rawBuffer;

    try {
      finalBuffer = await sharp(rawBuffer)
        .jpeg({ 
          quality: 92, 
          chromaSubsampling: '4:4:4', // Preserva fidelidade de cor
          mozjpeg: true               // Compressão eficiente sem perda visual
        })
        .toBuffer();
    } catch (sharpError) {
      console.warn('[SHARP OPTIMIZATION FAILED]', sharpError.message);
      // Fallback: retorna buffer original se a otimização falhar
    }

    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Content-Length', finalBuffer.length);
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    
    return res.status(200).send(finalBuffer);

  } catch (error) {
    console.error('[GENERATE ERROR]', error.message);
    
    if (error.name === 'AbortError' || error.message.includes('timeout')) {
      return res.status(504).json({ error: 'A geração expirou. Tente um prompt mais simples.' });
    }

    if (error.message.includes('Arquivo inválido')) {
      return res.status(400).json({ error: error.message });
    }

    return res.status(500).json({ error: 'Erro interno no servidor.' });
  }
});

app.listen(PORT, () => console.log(`[SERVER] API rodando na porta ${PORT}`));
