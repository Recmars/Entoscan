from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from .utils import load_labels, load_and_preprocess_image_keras

app = FastAPI()

# Load Keras model and labels
model = tf.keras.models.load_model("app/mobilenetv3-insect2.keras")
labels = load_labels("app/labels.txt")

# Get input shape from the model
input_shape = model.input_shape[1:3]

@app.post("/classify/")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image.save("temp.jpg")
        image_array = load_and_preprocess_image_keras("temp.jpg", input_shape)

        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = labels[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return JSONResponse({
            "predicted_class": predicted_class_label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
