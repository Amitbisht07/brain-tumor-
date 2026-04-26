import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import get_image_generators
from model_cnn import build_model
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(weights_path, train_dir, test_dir, image_size=(224,224), batch_size=16):
    print("✅ Building model architecture...")
    model = build_model(input_shape=(image_size[0], image_size[1], 3), num_classes=4)

    print("✅ Loading weights...")
    model.load_weights(weights_path)

    print("✅ Preparing test generator...")
    _, _, test_gen = get_image_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    print("✅ Predicting...")
    preds = model.predict(
        test_gen,
        steps=max(1, test_gen.samples // batch_size) + 1,
        verbose=1
    )

    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # ✅ Clean confusion matrix
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.3)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=1, linecolor='black', square=True
    )
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <weights_path>")
        sys.exit(1)

    weights_path = sys.argv[1]
    train_dir = "data/Training"
    test_dir = "data/Testing"

    evaluate(weights_path, train_dir, test_dir)
