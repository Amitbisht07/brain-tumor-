import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Reproducibility (same split/ordering every run)
SEED = 1337

def get_image_generators(
    train_dir: str,
    test_dir: str,
    image_size=(224, 224),
    batch_size: int = 16,
    val_split: float = 0.15,
):
    """
    Returns (train_gen, val_gen, test_gen) using EfficientNetB0 preprocessing.
    Folder structure must be:
        train_dir/
            ClassA/
            ClassB/
            ...
        test_dir/
            ClassA/
            ClassB/
            ...
    """

    # ---- TRAIN / VAL AUGMENTATION ----
    # NOTE: EfficientNet expects preprocess_input (NOT simple rescaling 1./255).
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split,   # split from train_dir
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.08,
        zoom_range=0.15,
        brightness_range=(0.9, 1.1),
        channel_shift_range=10.0,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # No augmentation on test; only EfficientNet preprocessing
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # ---- TRAIN GENERATOR ----
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
        interpolation="bilinear",
    )

    # ---- VALIDATION GENERATOR ----
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,                # keep deterministic order for metrics
        seed=SEED,
        interpolation="bilinear",
    )

    # ---- TEST GENERATOR ----
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,                # never shuffle at test time
        seed=SEED,
        interpolation="bilinear",
    )

    # Sanity print (helps you verify class order everywhere)
    print("Class indices (train):", train_gen.class_indices)

    return train_gen, val_gen, test_gen


def get_inference_datagen(image_size=(224, 224)):
    """
    Convenience helper: returns a single ImageDataGenerator configured for
    EfficientNetB0 preprocessing, useful if you build custom tf.data pipelines.
    """
    return ImageDataGenerator(preprocessing_function=preprocess_input)
