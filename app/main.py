from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from .utils import load_labels, load_and_preprocess_image, classify_image

app = FastAPI()

# Load TFLite model and labels
interpreter = tf.lite.Interpreter(model_path="mobilenetv3-insect3.tflite")
interpreter.allocate_tensors()
labels = load_labels("app/labels.txt")

# Get input shape from the model
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape'][1:3]  # Extract height and width

@app.post("/classify/")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image.save("temp.jpg") #Saving for temporary processing.
        image_array = load_and_preprocess_image("temp.jpg", input_shape) #Preprocess the image.

        predictions = classify_image(image_array, interpreter)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = labels[predicted_class_index]
        confidence = float(predictions[predicted_class_index])

        return JSONResponse({
            "predicted_class": predicted_class_label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
