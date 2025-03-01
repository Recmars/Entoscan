from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from .utils import load_labels, load_and_preprocess_image, classify_image

app = FastAPI()

# Load TFLite models and labels
tribe_interpreter = tf.lite.Interpreter(model_path="app/mobilenetv3-file2.tflite")
tribe_interpreter.allocate_tensors()
tribe_labels = load_labels("app/labels_tribe.txt")

insect_interpreter = tf.lite.Interpreter(model_path="app/mobilenetv3-insect3.tflite")
insect_interpreter.allocate_tensors()
insect_labels = load_labels("app/labels.txt")

# Get input shapes from the models
tribe_input_details = tribe_interpreter.get_input_details()
tribe_input_shape = tribe_input_details[0]['shape'][1:3]

insect_input_details = insect_interpreter.get_input_details()
insect_input_shape = insect_input_details[0]['shape'][1:3]

@app.post("/classify/")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image.save("temp.jpg")  # Saving for temporary processing.

        # Classify tribe
        tribe_image_array = load_and_preprocess_image("temp.jpg", tribe_input_shape)
        tribe_predictions = classify_image(tribe_image_array, tribe_interpreter)
        tribe_predicted_class_index = np.argmax(tribe_predictions)
        tribe_predicted_class_label = tribe_labels[tribe_predicted_class_index]
        tribe_confidence = float(tribe_predictions[tribe_predicted_class_index])

        if tribe_predicted_class_label == "batocerini insect":
            # Classify insect
            insect_image_array = load_and_preprocess_image("temp.jpg", insect_input_shape)
            insect_predictions = classify_image(insect_image_array, insect_interpreter)
            insect_predicted_class_index = np.argmax(insect_predictions)
            insect_predicted_class_label = insect_labels[insect_predicted_class_index]
            insect_confidence = float(insect_predictions[insect_predicted_class_index])

            return JSONResponse({
                "predicted_class": insect_predicted_class_label,
                "confidence": insect_confidence
            })
        else:
            return JSONResponse({
                "predicted_class": tribe_predicted_class_label,
                "confidence": tribe_confidence
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
