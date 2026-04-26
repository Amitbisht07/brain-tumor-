import os
import datetime
import numpy as np
import tensorflow as tf
from model_cnn import build_model
from data_generator import get_image_generators
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20


def compute_weights(train_gen):
    classes = train_gen.classes
    class_labels = np.unique(classes)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=classes
    )

    return dict(zip(class_labels, class_weights))


def main():
    print("✅ Loading data generators...")
    train_gen, val_gen, test_gen = get_image_generators(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print("✅ Computing class weights...")
    class_weights = compute_weights(train_gen)
    print("Class Weights:", class_weights)

    print("✅ Building EfficientNetB0 model...")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=4)

    # ---- CALLBACKS ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path = f"model/best_weights_{timestamp}.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=best_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            save_weights_only=True           # ✅ FIX
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]

    print("🚀 Training started...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Training complete. Best weights saved at:", best_path)

    final_path = f"model/final_weights_{timestamp}.h5"
    model.save_weights(final_path)
    print("✅ Final weights saved at:", final_path)


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    main()
