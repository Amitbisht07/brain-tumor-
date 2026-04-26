import os
import datetime
import numpy as np
import tensorflow as tf
from model_cnn import build_model
from data_generator import get_image_generators
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


# -------------------------------
# CONFIG
# -------------------------------
TRAIN_DIR   = "data/Training"
TEST_DIR    = "data/Testing"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16
FT_EPOCHS   = 12          # 10–15 is usually enough
UNFREEZE_TOP_N = 150      # unfreeze top 150 layers of the backbone
INITIAL_WEIGHTS = None    # set at runtime via CLI arg (best_weights_*.h5)


def compute_weights(train_gen):
    classes = train_gen.classes
    class_labels = np.unique(classes)
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=class_labels, y=classes)
    return dict(zip(class_labels, class_weights))


def freeze_batchnorm_layers(model):
    """Keep BatchNorm layers frozen during fine-tuning (stabilizes training)."""
    bn_frozen = 0
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            bn_frozen += 1
    print(f"✅ BatchNorm layers frozen: {bn_frozen}")


def unfreeze_top_layers(model, n=150):
    """
    Unfreeze the top 'n' layers of the base model; keep earlier layers frozen.
    Assumes the model was built via build_model() with EfficientNetB0 as base.
    """
    # First, unfreeze nothing
    for layer in model.layers:
        layer.trainable = False

    # Then unfreeze only the top N layers
    trainable_count = 0
    for layer in model.layers[-n:]:
        # skip BatchNorm here; we'll freeze them explicitly later
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
            trainable_count += 1

    print(f"✅ Unfroze top {n} layers (excluding BatchNorm). Trainable layers: {trainable_count}")


def count_params(model):
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    nontrainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"📦 Trainable params: {trainable:,} | Non-trainable params: {nontrainable:,}")


def main(initial_weights_path):
    print("✅ Loading data...")
    train_gen, val_gen, _ = get_image_generators(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print("✅ Computing class weights...")
    class_weights = compute_weights(train_gen)
    print("Class Weights:", class_weights)

    print("✅ Building model...")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=4)

    print(f"✅ Loading initial weights: {initial_weights_path}")
    model.load_weights(initial_weights_path)

    # ---- FINE-TUNE SETUP ----
    unfreeze_top_layers(model, n=UNFREEZE_TOP_N)
    freeze_batchnorm_layers(model)  # keep BN frozen
    count_params(model)

    # Low LR for fine-tuning
    optimizer = Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---- CALLBACKS ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path = f"model/best_finetuned_weights_{timestamp}.h5"

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
            save_weights_only=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            verbose=1,
            min_lr=1e-6
        ),
    ]

    print("🚀 Fine-tuning started...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FT_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fine-tuning complete. Best weights saved at:", best_path)

    final_path = f"model/final_finetuned_weights_{timestamp}.h5"
    model.save_weights(final_path)
    print("✅ Final fine-tuned weights saved at:", final_path)


if __name__ == "__main__":
    import sys
    os.makedirs("model", exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python finetune.py <initial_weights_path>")
        print("Example: python finetune.py model/best_weights_20251109-193555.h5")
        sys.exit(1)

    INITIAL_WEIGHTS = sys.argv[1]
    main(INITIAL_WEIGHTS)
