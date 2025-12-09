import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

def load_model():
    model = EfficientNetB0(weights="imagenet")
    return model

def predict_image(model, pil_img, top=3):
    img = pil_img.convert("RGB").resize((224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top)[0]

    results = [
        {"label": label, "probability": float(prob)}
        for (_, label, prob) in decoded
    ]
    return results

