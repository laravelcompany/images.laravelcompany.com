from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Force CPU-only operation at system level
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_CUDA_AVAILABLE'] = '0'

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import io
import base64
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image-to-Image Generator API (CPU Offline)", 
    version="1.0.0",
    description="CPU-optimized AI-powered image transformation API with offline model support"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the pipeline
pipeline = None

def load_pipeline():
    """Load the Stable Diffusion pipeline optimized for CPU with offline support"""
    global pipeline
    if pipeline is None:
        try:
            # Force CPU usage and disable CUDA completely
            device = "cpu"
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Ensure PyTorch uses CPU only
            torch.cuda.is_available = lambda: False
            
            logger.info(f"Loading model on device: {device}")
            
            cache_dir = "/home/trending/cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # First try to load from cache (offline mode)
            try:
                logger.info("Attempting to load model from cache (offline mode)...")
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    cache_dir=cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True  # Force offline loading
                )
                logger.info("Model loaded successfully from cache!")
                
            except Exception as offline_error:
                logger.warning(f"Failed to load from cache: {offline_error}")
                logger.info("Attempting to download model (online mode)...")
                
                # If offline loading fails, try downloading
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    cache_dir=cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=False  # Allow download
                )
                logger.info("Model downloaded and loaded successfully!")
            
            # Explicitly move all components to CPU
            pipeline = pipeline.to("cpu")
            
            # Ensure all model components are on CPU
            if hasattr(pipeline, 'unet'):
                pipeline.unet = pipeline.unet.to("cpu")
            if hasattr(pipeline, 'vae'):
                pipeline.vae = pipeline.vae.to("cpu")
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder = pipeline.text_encoder.to("cpu")
            if hasattr(pipeline, 'tokenizer'):
                # Tokenizer doesn't need .to() but ensure it exists
                pass
            
            # CPU-specific optimizations
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            
            # Set number of threads for CPU inference
            torch.set_num_threads(max(1, torch.get_num_threads() // 2))
                
            logger.info("Model pipeline configured successfully for CPU!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    return pipeline

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    try:
        logger.info("Starting CPU-optimized Image-to-Image API (Offline Mode)...")
        # Load the pipeline on startup for better performance
        load_pipeline()
        logger.info("Image-to-Image API started successfully with offline model support!")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        # Don't raise here to allow the API to start even if model loading fails
        logger.warning("API started but model loading failed - will retry on first request")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CPU-optimized Image-to-Image Generator API is running (Offline Mode)!",
        "service": "trending-img2img-api",
        "version": "1.0.0",
        "device": "cpu",
        "mode": "offline"
    }

@app.get("/health")
async def health_check():
    """Health check with model status"""
    global pipeline
    model_loaded = pipeline is not None
    
    # Check cache directory
    cache_dir = "/home/trending/cache"
    cache_exists = os.path.exists(cache_dir)
    cache_size = 0
    
    if cache_exists:
        try:
            # Calculate cache directory size
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    cache_size += os.path.getsize(filepath)
            cache_size_mb = cache_size / (1024 * 1024)  # Convert to MB
        except Exception:
            cache_size_mb = -1
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": "cpu",
        "mode": "offline",
        "cache_directory": cache_dir,
        "cache_exists": cache_exists,
        "cache_size_mb": round(cache_size_mb, 2) if cache_size_mb >= 0 else "unknown",
        "torch_version": torch.__version__,
        "cuda_available": False,
        "cpu_threads": torch.get_num_threads(),
        "service": "trending-img2img-api"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    global pipeline
    
    if pipeline is None:
        return {
            "model_loaded": False,
            "message": "Model not loaded yet"
        }
    
    return {
        "model_loaded": True,
        "model_id": "runwayml/stable-diffusion-v1-5",
        "device": "cpu",
        "mode": "offline",
        "components": {
            "unet": hasattr(pipeline, 'unet'),
            "vae": hasattr(pipeline, 'vae'),
            "text_encoder": hasattr(pipeline, 'text_encoder'),
            "tokenizer": hasattr(pipeline, 'tokenizer'),
            "scheduler": hasattr(pipeline, 'scheduler')
        }
    }

@app.post("/generate")
async def generate_image(
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt (optional)"),
    strength: float = Form(0.8, description="Strength of transformation (0.0-1.0)"),
    guidance_scale: float = Form(7.5, description="Guidance scale for generation"),
    num_inference_steps: int = Form(10, description="Number of inference steps (reduced for CPU)"),
    seed: Optional[int] = Form(None, description="Random seed (optional)")
):
    """
    Generate a new image based on input image and prompt (CPU-optimized, offline)
    """
    try:
        # Load pipeline if not already loaded
        pipe = load_pipeline()
        
        # Validate parameters (adjusted for CPU performance)
        if not 0.0 <= strength <= 1.0:
            raise HTTPException(status_code=400, detail="Strength must be between 0.0 and 1.0")
        
        if not 1.0 <= guidance_scale <= 15.0:
            raise HTTPException(status_code=400, detail="Guidance scale must be between 1.0 and 15.0")
        
        if not 1 <= num_inference_steps <= 50:  # Reduced max for CPU
            raise HTTPException(status_code=400, detail="Number of inference steps must be between 1 and 50")
        
        # Read and process the input image
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        
        # Resize image for CPU efficiency (smaller size)
        max_size = 512  # Reduced from 768 for faster CPU processing
        if max(input_image.size) > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Set up generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logger.info(f"Generating image with prompt: '{prompt[:50]}...' (CPU offline mode)")
        
        # Ensure we're using CPU for generation
        with torch.no_grad():  # Disable gradient computation for inference
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                generator=generator
            )
        
        # Get the generated image
        generated_image = result.images[0]
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        generated_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        logger.info("Image generated successfully on CPU (offline)")
        
        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/generate-base64")
async def generate_image_base64(
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt (optional)"),
    strength: float = Form(0.8, description="Strength of transformation (0.0-1.0)"),
    guidance_scale: float = Form(7.5, description="Guidance scale for generation"),
    num_inference_steps: int = Form(10, description="Number of inference steps (reduced for CPU)"),
    seed: Optional[int] = Form(None, description="Random seed (optional)")
):
    """
    Generate a new image and return as base64 encoded string (CPU-optimized, offline)
    """
    try:
        # Load pipeline if not already loaded
        pipe = load_pipeline()
        
        # Validate parameters
        if not 0.0 <= strength <= 1.0:
            raise HTTPException(status_code=400, detail="Strength must be between 0.0 and 1.0")
        
        # Read and process the input image
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        
        # Resize image for CPU efficiency
        max_size = 512
        if max(input_image.size) > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Set up generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logger.info(f"Generating base64 image with prompt: '{prompt[:50]}...' (CPU offline mode)")
        
        # Ensure we're using CPU for generation
        with torch.no_grad():  # Disable gradient computation for inference
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                generator=generator
            )
        
        # Get the generated image
        generated_image = result.images[0]
        
        # Convert to base64
        img_buffer = io.BytesIO()
        generated_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        logger.info("Base64 image generated successfully on CPU (offline)")
        
        return {
            "image": img_base64,
            "prompt": prompt,
            "parameters": {
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "device": "cpu",
                "mode": "offline"
            },
            "service": "trending-img2img-api"
        }
        
    except Exception as e:
        logger.error(f"Error generating base64 image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1054)