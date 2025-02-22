import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# Load TFLite model using the Interpreter
interpreter = tf.lite.Interpreter(model_path="model/mobilenetv3-classification3.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize to the model's input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

class_labels = [
    "Abatocera_leonina", "Abatocera_luzonica", "Apriona_jirouxi",
    "Apriona_rixato", "Batocera_magica", "Batocera_rubus", "Batocera_victoriana"
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    # Retrieve the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    
    return {
        "class": class_labels[predicted_index],
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)