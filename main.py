from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Define the maximum file size (e.g., 10 MB)
MAX_FILE_SIZE_MB = 10  # 10 MB
MAX_IMAGE_SIZE = (500, 500)  # Resize to 500x500

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Check file size limit
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds the 10 MB limit")

        # Open image using Pillow to resize if necessary
        image = Image.open(io.BytesIO(contents))

        # Resize the image to ensure it's within acceptable size
        image = image.resize(MAX_IMAGE_SIZE)

        # Convert the image to RGB (in case it's in a different mode, e.g., grayscale)
        image = image.convert("RGB")
        
        # Convert image to a numpy array and use OpenCV
        img = np.array(image)

        # Ensure the image is in the correct color format (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Analyze emotion, age, gender using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Convert emotion scores from float32 to native float
        emotion_scores = {k: float(v) for k, v in result[0]['emotion'].items()}

        return JSONResponse({
            "dominant_emotion": str(result[0]['dominant_emotion']),
            "emotion_scores": emotion_scores
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
