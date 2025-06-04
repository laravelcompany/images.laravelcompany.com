Here's a comprehensive `README.md` for your FastAPI-based CPU-optimized image-to-image generation API:

---

# 🖼️ CPU-Optimized Image-to-Image Generator API (Offline Mode)

This project provides an AI-powered image-to-image generation API using the [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) model optimized for **CPU-only environments** with **offline support**.

Built with **FastAPI**, it allows transforming images based on user-provided prompts, suitable for local, low-resource deployments or edge systems.

---

## 🚀 Features

* ✅ CPU-only inference (no GPU required)
* ✅ Offline mode using cached Hugging Face models
* ✅ FastAPI with OpenAPI/Swagger UI docs
* ✅ CORS-enabled
* ✅ Input image resizing for CPU efficiency
* ✅ Two generation modes: downloadable image or base64 JSON

---

## 📦 Requirements

* Python 3.8+
* PyTorch (CPU version)
* `diffusers`, `transformers`, `fastapi`, `uvicorn`, `Pillow`

Install dependencies:

```bash
pip install -r requirements.txt
```

*Example `requirements.txt`:*

```txt
torch==2.1.0  # or latest CPU-compatible version
diffusers
transformers
fastapi
uvicorn
pillow
```

---

## 🛠️ Usage

### 🔧 Run the API

```bash
python app.py
```

The API will be available at `http://localhost:1054`.

Swagger docs: `http://localhost:1054/docs`
ReDoc docs: `http://localhost:1054/redoc`

---

## 🧠 Model Details

* **Model ID**: `runwayml/stable-diffusion-v1-5`
* **Cache Directory**: `/home/trending/cache`
* First attempts to load from local cache (offline mode)
* Falls back to online download if needed (can be disabled)

---

## 📂 Endpoints

### `GET /`

Simple health check.

### `GET /health`

Full health and model/cache status.

### `GET /model-info`

Returns detailed info about loaded components.

---

### `POST /generate`

**Generate a new image and return a downloadable PNG**

**Form fields:**

| Field                 | Type    | Description                          |
| --------------------- | ------- | ------------------------------------ |
| `image`               | file    | Input image file (e.g. PNG/JPG)      |
| `prompt`              | string  | Text prompt                          |
| `negative_prompt`     | string? | Negative prompt (optional)           |
| `strength`            | float   | Image transformation strength (0-1)  |
| `guidance_scale`      | float   | Prompt guidance scale (default: 7.5) |
| `num_inference_steps` | int     | Inference steps (1–50 recommended)   |
| `seed`                | int?    | Random seed (optional)               |

---

### `POST /generate-base64`

**Same as `/generate`, but returns image as a base64 string in JSON**

---

## 🧪 Example with `curl`

```bash
curl -X POST http://localhost:1054/generate \
  -F "image=@input.png" \
  -F "prompt=A fantasy castle on a hill" \
  --output output.png
```

Or for base64:

```bash
curl -X POST http://localhost:1054/generate-base64 \
  -F "image=@input.png" \
  -F "prompt=A glowing dragon flying through clouds"
```

---

## 🧾 Environment Behavior

* Forces CPU-only use by setting:

  ```python
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.environ['TORCH_CUDA_AVAILABLE'] = '0'
  ```
* Thread usage optimized:

  ```python
  torch.set_num_threads(max(1, torch.get_num_threads() // 2))
  ```

---

## 📌 Notes

* Designed for **low-spec machines** and **offline environments**
* Does not require internet after initial model download (unless cache is missing)
* Initial model load may take several minutes on first run
* Compatible with Hugging Face offline cache system

---

## 📄 License

MIT License

---

## 🙏 Credits

* [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* [FastAPI](https://fastapi.tiangolo.com/)
* `runwayml/stable-diffusion-v1-5` model

* [Laravel Development Agency](https://laravelcompany.com/)