const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");
const Busboy = require("busboy");
const sharp = require("sharp");

const app = express();
app.use(cors());

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const GEN_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b";
const EDIT_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev";

if (!NVIDIA_API_KEY) console.warn("[WARN] NVIDIA_API_KEY não definida.");

function parseForm(req) {
  return new Promise((resolve, reject) => {
    const bb = Busboy({ headers: req.headers, limits: { fileSize: 20 * 1024 * 1024 } });
    const fields = {}, files = {};
    bb.on("file", (name, stream, info) => {
      const chunks = [];
      stream.on("data", (c) => chunks.push(c));
      stream.on("end", () => { files[name] = { buffer: Buffer.concat(chunks), ...info }; });
    });
    bb.on("field", (name, val) => { fields[name] = val; });
    bb.on("close", () => resolve({ fields, files }));
    bb.on("error", reject);
    req.pipe(bb);
  });
}

app.get("/", (req, res) => {
  res.json({ status: "ok", genUrl: GEN_URL, editUrl: EDIT_URL });
});

app.post("/generate", async (req, res) => {
  try {
    const { fields, files } = await parseForm(req);
    const prompt = (fields.param || fields.prompt || "").trim();

    if (!prompt) {
      return res.status(400).json({ error: "Preciso de uma descrição pra criar algo. Me conta o que você quer ver!" });
    }

    const hasImage = !!(files.image && files.image.buffer && files.image.buffer.length > 0);

    let nvidiaRes;
    if (hasImage) {
      const dataUri = `data:${files.image.mimeType || "image/jpeg"};base64,${files.image.buffer.toString("base64")}`;
      nvidiaRes = await fetch(EDIT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: `Bearer ${NVIDIA_API_KEY}`,
        },
        body: JSON.stringify({
          prompt,
          image: dataUri,
          samples: 1,
          seed: 0,
          steps: 30,
        }),
      });
    } else {
      nvidiaRes = await fetch(GEN_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: `Bearer ${NVIDIA_API_KEY}`,
        },
        body: JSON.stringify({
          prompt,
          samples: 1,
          seed: 0,
          steps: 4,
        }),
      });
    }

    const rawText = await nvidiaRes.text();
    let data;
    try {
      data = JSON.parse(rawText);
    } catch {
      console.error("[NVIDIA NON-JSON]", nvidiaRes.status, rawText.slice(0, 500));
      return res.status(500).json({
        error: "Não consegui criar a imagem desta vez. Pode tentar de novo?",
        nvidia_status: nvidiaRes.status,
        nvidia_raw: rawText.slice(0, 500),
      });
    }

    if (!nvidiaRes.ok) {
      console.error("[NVIDIA ERROR]", nvidiaRes.status, JSON.stringify(data));
      if (nvidiaRes.status === 429) {
        return res.status(429).json({ error: "Estou com muita demanda agora, precisei parar pra respirar. Aguarda um pouquinho e tenta de novo? 🖤" });
      }
      return res.status(500).json({
        error: "Não consegui criar a imagem desta vez. Pode tentar de novo?",
        nvidia_status: nvidiaRes.status,
        nvidia_details: data,
      });
    }

    // formato NIM: { artifacts: [{ base64: "..." }] }
    const b64 = data?.artifacts?.[0]?.base64 || data?.data?.[0]?.b64_json;
    if (!b64) {
      console.error("[NVIDIA NO-IMAGE]", JSON.stringify(data).slice(0, 500));
      return res.status(500).json({
        error: "A imagem sumiu antes de chegar até você. Tenta de novo!",
        nvidia_details: data,
      });
    }

    const raw = Buffer.from(b64, "base64");

    let final = raw;
    try {
      final = await sharp(raw)
        .sharpen({ sigma: 1.2, m1: 1.5, m2: 0.7, x1: 2, y2: 10, y3: 20 })
        .modulate({ brightness: 1.02, saturation: 1.12 })
        .jpeg({ quality: 98, chromaSubsampling: "4:4:4", mozjpeg: true })
        .toBuffer();
    } catch (e) {
      console.warn("[POST-PROCESS WARN]", e.message);
    }

    res.setHeader("Content-Type", "image/jpeg");
    res.setHeader("Content-Length", final.length);
    res.setHeader("Cache-Control", "no-cache");
    return res.status(200).send(final);
  } catch (err) {
    console.error("[GENERATE ERROR]", err.message, err.stack);
    if (err.type === "request.aborted" || err.message?.includes("timeout")) {
      return res.status(504).json({ error: "Fiquei tanto tempo pensando que me perdi... Tenta de novo com uma ideia mais simples?" });
    }
    return res.status(500).json({
      error: "Não consegui criar a imagem desta vez. Pode tentar de novo?",
      debug: err.message,
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`API rodando na porta ${PORT}`));
