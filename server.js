const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const Busboy = require('busboy');
const sharp = require('sharp');

const app = express();

// ============================================================================
// CONFIGURAÇÕES E CONSTANTES
// ============================================================================
const PORT = process.env.PORT || 3000;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

// Modelo principal: Stable Diffusion 3 Medium (entende português!)
const SD3_URL = 'https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium';

// Fallback para edição de imagem (se precisar)
const KONTEDIT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const NVIDIA_TIMEOUT = 180000; // 3 minutos (SD3 pode ser mais lento)
const MAX_RETRIES = 2;

// Parâmetros otimizados para SD3 Medium
const SD3_PARAMS = {
  aspect_ratio: '1:1',      // Pode ser: 1:1, 16:9, 9:16, 5:4, 4:5, 3:2, 2:3
  cfg_scale: 7,              // 5-9: quanto maior, mais fiel ao prompt
  mode: 'text-to-image',
  model: 'sd3',
  output_format: 'jpeg',
  seed: 0,                   // 0 = aleatório
  steps: 50,                 // 50-100 para melhor qualidade
};

// Negative prompt universal para evitar deformações
const UNIVERSAL_NEGATIVE_PROMPT = [
  'deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy',
  'extra limb, missing limb, floating limbs, mutated hands and fingers',
  'disconnected limbs, mutation, ugly, disgusting, blurry, amputation',
  'duplicate, morbid, mutilated, out of frame, extra fingers',
  'badly drawn hands, badly drawn face, mutation, deformed',
  'watermark, signature, text, logo, banner, extra digits',
  'cropped, worst quality, low quality, jpeg artifacts'
].join(', ');

if (!NVIDIA_API_KEY) {
  console.error('[FATAL] NVIDIA_API_KEY não definida no ambiente.');
}

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ============================================================================
// UTILITÁRIOS
// ============================================================================

function parseFormData(req) {
  return new Promise((resolve, reject) => {
    const busboy = Busboy({ 
      headers: req.headers, 
      limits: { fileSize: MAX_FILE_SIZE, files: 1, fields: 10 },
      defCharset: 'utf8'
    });
    
    const fields = {};
    const files = {};
    let fileCount = 0;

    busboy.on('file', (fieldname, stream, info) => {
      fileCount++;
      
      if (fileCount > 1) {
        stream.resume();
        return reject(new Error('Múltiplos arquivos não permitidos.'));
      }
      
      if (!ALLOWED_MIME_TYPES.includes(info.mimeType)) {
        stream.resume();
        return reject(new Error(`Tipo de arquivo inválido: ${info.mimeType}`));
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

        if (response.status < 500 || attempt === MAX_RETRIES) {
          return response;
        }

        console.warn(`[NVIDIA RETRY] Status ${response.status}, tentativa ${attempt}/${MAX_RETRIES}`);
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      } catch (err) {
        if (err.name === 'AbortError') {
          throw new Error('Timeout na comunicação com NVIDIA NIM.');
        }
        if (attempt === MAX_RETRIES) throw err;
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      }
    }
  } finally {
    clearTimeout(timeout);
  }
}

// ============================================================================
// ROTAS
// ============================================================================

app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Bunix Image API',
    version: '3.0',
    model: 'Stable Diffusion 3 Medium (Multilíngue)',
    endpoints: { generate: '/generate' }
  });
});

app.post('/generate', async (req, res) => {
  try {
    const { fields, files } = await parseFormData(req);
    const prompt = (fields.param || fields.prompt || '').trim();

    if (!prompt) {
      return res.status(400).json({ 
        error: 'Prompt obrigatório. Descreva o que você quer ver.' 
      });
    }

    if (prompt.length > 10000) {
      return res.status(400).json({ 
        error: 'Prompt muito longo. Máximo 10000 caracteres.' 
      });
    }

    const hasImage = files.image && files.image.buffer && files.image.buffer.length > 0;
    let nvidiaResponse;
    let requestBody;

    // =========================================================================
    // GERAÇÃO COM SD3 MEDIUM (Entende português!)
    // =========================================================================
    
    // Aspect ratio inteligente baseado na imagem de referência (se houver)
    let aspectRatio = '1:1';
    if (hasImage) {
      try {
        const metadata = await sharp(files.image.buffer).metadata();
        const ratio = metadata.width / metadata.height;
        
        if (ratio > 1.7) aspectRatio = '16:9';
        else if (ratio > 1.2) aspectRatio = '5:4';
        else if (ratio < 0.55) aspectRatio = '9:16';
        else if (ratio < 0.8) aspectRatio = '4:5';
        else if (ratio < 1.1) aspectRatio = '1:1';
        else aspectRatio = '3:2';
      } catch (e) {
        console.warn('[ASPECT RATIO DETECT FAILED]', e.message);
      }
    }

    // Customizações opcionais via fields
    const customCfg = parseFloat(fields.cfg_scale) || SD3_PARAMS.cfg_scale;
    const customSteps = parseInt(fields.steps) || SD3_PARAMS.steps;
    const customSeed = parseInt(fields.seed) || SD3_PARAMS.seed;
    const customNegative = fields.negative_prompt || UNIVERSAL_NEGATIVE_PROMPT;

    requestBody = {
      prompt: prompt,
      negative_prompt: customNegative,
      aspect_ratio: fields.aspect_ratio || aspectRatio,
      cfg_scale: Math.min(Math.max(customCfg, 1), 9),
      mode: 'text-to-image',
      model: 'sd3',
      output_format: 'jpeg',
      seed: customSeed,
      steps: Math.min(Math.max(customSteps, 5), 100),
    };

    nvidiaResponse = await requestNvidia(SD3_URL, requestBody);

    // =========================================================================
    // PROCESSA A RESPOSTA
    // =========================================================================
    const rawText = await nvidiaResponse.text();
    let data;
    
    try {
      data = JSON.parse(rawText);
    } catch {
      console.error('[NVIDIA PARSE ERROR]', nvidiaResponse.status, rawText.slice(0, 500));
      return res.status(500).json({ 
        error: 'Resposta inválida do provedor de IA. Tente novamente.' 
      });
    }

    if (!nvidiaResponse.ok) {
      if (nvidiaResponse.status === 429) {
        return res.status(429).json({ 
          error: 'Muitas requisições simultâneas. Aguarde 30 segundos e tente novamente.' 
        });
      }
      if (nvidiaResponse.status === 422) {
        return res.status(422).json({ 
          error: 'Prompt bloqueado pelo filtro de segurança. Reformule sua descrição.',
          details: data 
        });
      }
      return res.status(nvidiaResponse.status || 500).json({
        error: 'Falha ao gerar a imagem.',
        nvidia_status: nvidiaResponse.status,
        nvidia_details: data,
      });
    }

    // Extrai a imagem
    const artifact = data?.artifacts?.[0];
    const base64Image = artifact?.base64 || data?.data?.[0]?.b64_json || data?.image;

    if (artifact?.finishReason === 'CONTENT_FILTERED' || !base64Image) {
      return res.status(422).json({
        error: 'Conteúdo bloqueado pelas políticas de segurança.',
        nvidia_details: data,
      });
    }

    // Otimização pura (sem deformações)
    const rawBuffer = Buffer.from(base64Image, 'base64');
    let finalBuffer = rawBuffer;

    try {
      finalBuffer = await sharp(rawBuffer)
        .jpeg({ 
          quality: 92, 
          chromaSubsampling: '4:4:4',
          mozjpeg: true,
          progressive: true
        })
        .toBuffer();
    } catch (sharpError) {
      console.warn('[SHARP OPTIMIZATION FAILED]', sharpError.message);
    }

    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Content-Length', finalBuffer.length);
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    
    return res.status(200).send(finalBuffer);

  } catch (error) {
    console.error('[GENERATE ERROR]', error.message, error.stack);
    
    if (error.name === 'AbortError' || error.message.includes('timeout')) {
      return res.status(504).json({ 
        error: 'A geração demorou muito. Tente um prompt mais simples ou reduza os steps.' 
      });
    }

    if (error.message.includes('inválido') || error.message.includes('permitidos')) {
      return res.status(400).json({ error: error.message });
    }

    return res.status(500).json({ 
      error: 'Erro interno no servidor. Tente novamente em alguns segundos.' 
    });
  }
});

app.listen(PORT, () => {
  console.log(`[SERVER] API rodando na porta ${PORT}`);
  console.log(`[SERVER] Modelo: Stable Diffusion 3 Medium (Multilíngue)`);
  console.log(`[SERVER] Endpoint: /generate`);
});
