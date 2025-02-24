import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from fastapi import FastAPI, UploadFile, File
import uvicorn
import traceback

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
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {
            "class": class_labels[predicted_index],
            "confidence": confidence
        }
    except UnidentifiedImageError:
        return {"error": "Uploaded file is not a valid image."}, 400 #Bad Request.
    except Exception as e:
        error_message = f"Error during prediction: {e}\n{traceback.format_exc()}"
        print(error_message)
        return {"error": error_message}, 500

if __name__ == "__main__":
    print("Starting server on 0.0.0.0:8000") #add print statement.
    uvicorn.run(app, host="0.0.0.0", port=8000)
