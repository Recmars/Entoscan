from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io

app = FastAPI()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image to match the TFLite model input requirements."""
    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handle image upload and process with TFLite model."""
    image = Image.open(io.BytesIO(await file.read()))
    image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return {"predictions": output_data.tolist()}
