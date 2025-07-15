# ---------------------------------------------
# Credits:
# - Some logic and inspiration for this project were assisted by Claude AI.
# - Additional guidance and code patterns were adapted from a YouTube tutorial.
# ---------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import os
from model_manager import PostModelManager  # Import the class

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

# Input models
class DuplicateCheckInput(BaseModel):
    user_id: str = None
    title: str
    body: str

class TaggingInput(BaseModel):
    post_id: str
    post_body: str

# Output models
class DuplicateCheckOutput(BaseModel):
    is_duplicate: bool
    similar_post_id: str
    similarity_score: float
    status: str

class TaggingOutput(BaseModel):
    post_id: str
    tags: list[str]

# Global variable to store model
model = None

def load_model():
    """Load the model if not already loaded"""
    global model
    if model is None:
        model_file = "post_model.pkl"
        
        # Check if model file exists
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found. Creating new model...")
            # Create model if it doesn't exist
            try:
                model = PostModelManager()
                model.build_from_files('posts_dataset.json', 'config.json')
                model.save_model(model_file)
                print("New model created and saved successfully!")
            except Exception as e:
                print(f"Error creating new model: {e}")
                raise HTTPException(status_code=500, detail="Failed to create model")
        else:
            print("Loading existing model...")
            try:
                # Use the PostModelManager.load_model() method instead of pickle.load()
                model = PostModelManager.load_model(model_file)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting to create new model...")
                try:
                    model = PostModelManager()
                    model.build_from_files('posts_dataset.json', 'config.json')
                    model.save_model(model_file)
                    print("New model created and saved successfully!")
                except Exception as e2:
                    print(f"Error creating new model: {e2}")
                    raise HTTPException(status_code=500, detail="Failed to load or create model")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        config_version = config.get("version", "1.0")
    except Exception:
        config_version = "1.0"
    model_version = "1.0"
    return {"model_version": model_version, "config_version": config_version}

@app.post("/check-duplicate", response_model=DuplicateCheckOutput)
def check_duplicate(input: DuplicateCheckInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to the format expected by your model
        input_post = {
            'title': input.title,
            'body': input.body
        }
        
        # Use your pre-built model
        result = model.check_duplicate_post(input_post)
        return DuplicateCheckOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking duplicate: {str(e)}")

@app.post("/generate-tags", response_model=TaggingOutput)
def generate_tags(input: TaggingInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use your pre-built model for tag generation
        tags = model.generate_post_tags(input.post_body)
        return TaggingOutput(post_id=input.post_id, tags=tags)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tags: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)