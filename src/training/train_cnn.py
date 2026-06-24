import os
import mlflow
import tensorflow as tf
import numpy as np
from config import settings
from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import get_cnn_datasets, get_dataset_version
from src.mlops.mlflow_utils import setup_mlflow, log_run_metrics


def build_cnn_model(img_size=(224, 224), num_classes=39, l2_reg=1e-4, dropout_rate=0.4):
    regularizer = tf.keras.regularizers.l2(l2_reg)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),

        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),

        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(name="embedding_latent"),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate / 2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Keras callback- Logguea métricas por época a MLFLOW
# ──────────────────────────────────────────────────────────────────────────────
class MLflowEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        mlflow.log_metrics({
            "epoch_train_loss":     logs.get("loss", 0),
            "epoch_train_accuracy": logs.get("accuracy", 0),
            "epoch_val_loss":       logs.get("val_loss", 0),
            "epoch_val_accuracy":   logs.get("val_accuracy", 0),
        }, step=epoch)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline de Entrenamiento
# ──────────────────────────────────────────────────────────────────────────────
def train_cnn_pipeline(config_path="config/train_config.yaml", quick_train=False):
    # Si se añade "--quick", se utiliza una menor cantidad de datos/imágenes del dataset
    config = load_yaml_config(config_path)

    setup_mlflow(config)

    zip_path = os.path.join(settings.DATA_DIR, "raw", "Plant_leaf_diseases_dataset_with_augmentation.zip")
    dataset_version = get_dataset_version(zip_path)

    cnn_params = config['cnn']
    img_size    = tuple(cnn_params['img_size'])
    batch_size  = cnn_params['batch_size']
    val_split   = cnn_params['validation_split']
    seed        = cnn_params['seed']
    patience    = cnn_params.get('patience', 5)
    epochs      = 1 if quick_train else cnn_params['epochs']
    lr          = cnn_params['learning_rate']

    try:
        train_dataset, val_dataset, class_names = get_cnn_datasets(
            data_dir=settings.RAW_DATA_DIR,
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split,
            seed=seed
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    num_classes = len(class_names)
    print(f"[INFO] Dataset cargado. Total de clases: {num_classes}")

    if quick_train:
        print("[INFO] Quick-train activado. Reduciendo dataset a unos pocos batches...")
        train_dataset = train_dataset.take(5)
        val_dataset   = val_dataset.take(2)

    run_name = config['mlflow']['cnn_run_name']

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[INFO] Iniciando run de MLflow para CNN: {run_name}")

        model = build_cnn_model(
            img_size=img_size,
            num_classes=num_classes,
            l2_reg=cnn_params['l2_reg'],
            dropout_rate=cnn_params['dropout_rate']
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            MLflowEpochLogger(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(2, patience // 2),
                min_lr=1e-6,
                verbose=1
            ),
        ]

        print(f"[INFO] Iniciando entrenamiento por hasta {epochs} épocas...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        print("[INFO] Entrenamiento finalizado. Obteniendo predicciones de validación...")

        y_true, y_pred_list = [], []
        for images, labels in val_dataset:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred_list.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred_list)

        parameters = {
            "img_size":      str(img_size),
            "batch_size":    batch_size,
            "epochs":        len(history.history['loss']),
            "learning_rate": lr,
            "l2_reg":        cnn_params['l2_reg'],
            "dropout_rate":  cnn_params['dropout_rate'],
            "patience":      patience,
            "architecture":  "CNN_4block_BN_augment"
        }

        log_run_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            dataset_version=dataset_version,
            parameters=parameters,
            config_dict=config,
            stage_name="train_cnn",
            model=model,
            model_type="keras"
        )

        # ── Guardar el modelo localmente ──
        os.makedirs(os.path.join(settings.BASE_DIR, "models"), exist_ok=True)
        model_save_path = os.path.join(settings.BASE_DIR, "models", "cnn_model.keras")
        model.save(model_save_path)
        print(f"[INFO] Modelo CNN guardado localmente en: {model_save_path}")


if __name__ == "__main__":
    train_cnn_pipeline()
