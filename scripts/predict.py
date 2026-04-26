import sys
import numpy as np
import cv2
from model_cnn import build_model
from tensorflow.keras.applications.efficientnet import preprocess_input

CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]


def predict_image(weights_path, image_path):
    print("✅ Building model...")
    model = build_model(input_shape=(224, 224, 3), num_classes=4)

    print("✅ Loading weights...")
    model.load_weights(weights_path)

    print("✅ Loading image:", image_path)
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)

    # ✅ probabilities must be printed AFTER preds is defined
    print("\nRaw probabilities:", preds[0])

    class_id = np.argmax(preds)
    confidence = preds[0][class_id] * 100

    print("\n-----------------------------------")
    print(f"🧠 Predicted Class: {CLASS_LABELS[class_id]}")
    print(f"✅ Confidence: {confidence:.2f}%")
    print("-----------------------------------")

    return CLASS_LABELS[class_id], confidence


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <weights_path> <image_path>")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_path = sys.argv[2]

    predict_image(weights_path, image_path)
