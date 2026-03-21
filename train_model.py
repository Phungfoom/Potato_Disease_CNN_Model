import os
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tqdm import tqdm

import config
from Potato_bunch_brain import build_combined_model
from grad_cam_visualizer import explain_my_model

# Leaf-only training; field testing handled by separate scripts.
parser = argparse.ArgumentParser(description="Train Potato ID Fusion Model (leaf-only)")
parser.add_argument('--stage', type=str, default="foundation", choices=['foundation', 'midterm'])
parser.add_argument(
    '--augment',
    action='store_true',
    help='Enable training-only data augmentation (recommended for imbalanced classes).',
)
parser.add_argument(
    '--validation-freq',
    type=int,
    default=2,
    metavar='N',
    help=(
        'Run Keras validation every N epochs (default: 2). '
        'Use 1 for every epoch; larger N speeds up training but sparser val_* in history plots.'
    ),
)
args = parser.parse_args()
if args.validation_freq < 1:
    raise SystemExit("--validation-freq must be >= 1")


def _print_device_summary() -> None:
    """Show whether TensorFlow sees a GPU (CPU-only training is usually much slower)."""
    print(f"\n[TensorFlow] version: {tf.__version__}")
    try:
        cpus = tf.config.list_physical_devices("CPU")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[TensorFlow] CPUs: {len(cpus)}  GPUs: {len(gpus)}")
        for d in gpus:
            dtype = getattr(d, "device_type", "?")
            print(f"  GPU: {d.name} ({dtype})")
        if gpus:
            print(
                "[TensorFlow] At least one GPU is visible — Keras will use it for "
                "supported ops (see Task Manager / nvidia-smi while training to confirm load)."
            )
        else:
            print(
                "[TensorFlow] No GPU visible — training will use CPU only "
                "(install CUDA/cuDNN build of TF + drivers if you expect a GPU)."
            )
    except Exception as exc:  # pragma: no cover
        print(f"[TensorFlow] Could not list devices: {exc}")


_print_device_summary()

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = "leaf"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _tf_cache_paths(
    *,
    script_dir: str,
    stage: str,
    augment: bool,
) -> tuple[str, str, str]:
    """
    Per-stage subfolders + pipeline-specific basename so caches do not collide or go stale silently.

    Layout:
      outputs/tf_cache/<stage>/train/<basename>.*
      outputs/tf_cache/<stage>/val/<basename>.*
    """
    h, w = config.DATA_PARAMS["image_size"]
    bs = config.DATA_PARAMS["batch_size"]
    pipe = getattr(config, "TF_CACHE_PIPELINE_ID", "v1")
    basename = (
        f"spud_{pipe}_bs{bs}_h{h}x{w}_aug{int(augment)}"
    )
    root = os.path.join(script_dir, config.OUTPUT_DIR, "tf_cache", stage)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    train_prefix = os.path.join(train_dir, basename)
    val_prefix = os.path.join(val_dir, basename)
    return train_prefix, val_prefix, basename


_train_cache_prefix, _val_cache_prefix, _cache_basename = _tf_cache_paths(
    script_dir=script_dir,
    stage=args.stage,
    augment=bool(args.augment),
)
print(
    f"[tf.data cache] stage={args.stage} basename={_cache_basename}\n"
    f"  train -> {_train_cache_prefix}.*\n"
    f"  val   -> {_val_cache_prefix}.*"
)

# Train on foundation, then midterm for fields, and NASA API for final check
base = os.path.join(script_dir, config.BASE_DIR)
if args.stage == "foundation":
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")
    # Use pre-split train/val from data_preprocessing (stratified by area and class) when present
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        data_dir = train_dir
        train_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="rgb",
            shuffle=True,
            **config.DATA_PARAMS)
        val_raw = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            color_mode="rgb",
            shuffle=False,
            **config.DATA_PARAMS)
    else:
        data_dir = os.path.join(base, f'{DOMAIN}_classes')
        train_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            color_mode="rgb",
            shuffle=True,
            **config.DATA_PARAMS)
        val_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            color_mode="rgb",
            shuffle=True,
            **config.DATA_PARAMS)
else:
    # Midterm uses a different directory for "Field" tests
    data_dir = os.path.join(base, f'midterm_{DOMAIN}_data')
    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        color_mode="rgb",
        shuffle=True,
        **config.DATA_PARAMS)
    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        color_mode="rgb",
        shuffle=True,
        **config.DATA_PARAMS)

# Compute class weights to handle imbalanced datasets
y_train = np.concatenate([y for _, y in train_raw], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
class_weights_dict = dict(enumerate(weights))

class_names = train_raw.class_names
num_classes = len(class_names)

val_labels = np.concatenate([y for _, y in val_raw], axis=0)
print("Validation images per class:", dict(zip(class_names, np.bincount(val_labels))))

# Training-only augmentation to improve minority-class generalization.
# This is applied ONLY on the training pipeline (never on validation / field testing).
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ],
    name="data_augmentation",
)


def _to_triple(norm_rgb: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gray = tf.image.rgb_to_grayscale(norm_rgb)
    return norm_rgb, gray, gray


def prepare_triple_input_train(image, label):
    """Convert RGB image to normalized triple with optional augmentation."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    if args.augment:
        # `image_dataset_from_directory` already yields batched tensors.
        aug = data_augmentation(norm_image, training=True)
        # Brightness is easier as an image op (kept mild to avoid label corruption).
        aug = tf.image.random_brightness(aug, max_delta=0.10)
        norm_image = tf.clip_by_value(aug, 0.0, 1.0)
    rgb, gray, sobel = _to_triple(norm_image)
    return (rgb, gray, sobel), label


def prepare_triple_input_eval(image, label):
    """Convert RGB image to normalized triple without augmentation."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    rgb, gray, sobel = _to_triple(norm_image)
    return (rgb, gray, sobel), label


train_dataset = (
    train_raw.map(prepare_triple_input_train, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(_train_cache_prefix)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(_val_cache_prefix)
    .prefetch(tf.data.AUTOTUNE)
)


def collect_val_predictions(model, val_ds):
    y_true_chunks, y_pred_chunks = [], []
    for images, labels in tqdm(val_ds, desc="Val predictions (reports/plots)"):
        preds = model.predict_on_batch(images)
        y_true_chunks.append(labels.numpy())
        y_pred_chunks.append(np.argmax(preds, axis=1))
    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    return y_true, y_pred


def plot_training_history(history, plot_dir, domain, stage, timestamp):
    """Plot training and validation accuracy/loss curves side by side."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    h = history.history
    n_train = len(h['accuracy'])
    train_x = range(n_train)

    # Accuracy plot
    axes[0].plot(train_x, h['accuracy'], label='Train Acc')
    va = h.get('val_accuracy')
    if va is not None and len(va) == n_train:
        axes[0].plot(train_x, va, label='Val Acc')
    elif va is not None:
        # e.g. validation_freq > 1: Keras may record fewer val_* points than epochs
        axes[0].plot(range(len(va)), va, label='Val Acc (sparse)', marker='o')
    axes[0].set_title(f'Accuracy: {domain} ({stage})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(train_x, h['loss'], label='Train Loss')
    vl = h.get('val_loss')
    if vl is not None and len(vl) == n_train:
        axes[1].plot(train_x, vl, label='Val Loss')
    elif vl is not None:
        axes[1].plot(range(len(vl)), vl, label='Val Loss (sparse)', marker='o')
    axes[1].set_title(f'Loss: {domain} ({stage})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'history_{domain}_{stage}_{timestamp}.png'))
    plt.close()

model = build_combined_model(num_classes=num_classes)

# Set up output directories
model_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'models')
plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'plots')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.EPOCHS,
    class_weight=class_weights_dict,
    validation_freq=args.validation_freq,
)

model.save(os.path.join(model_dir, f'potato_{DOMAIN}_model_{timestamp}.keras'))
plot_training_history(history, plot_dir, DOMAIN, args.stage, timestamp)

# Per-sample preds for sklearn (see collect_val_predictions docstring).
y_true, y_pred = collect_val_predictions(model, val_dataset)


def save_report_as_image(y_true, y_pred, target_names, plot_dir, timestamp):
    """Save classification report as an image table."""
    # Get only classes that appear in the validation set
    present_classes = np.unique(y_true)
    filtered_names = [target_names[i] for i in present_classes]

    # Create the report as a dictionary and convert to DataFrame
    report = classification_report(
        y_true, y_pred, 
        labels=present_classes, 
        target_names=filtered_names, 
        output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    plt.figure(figsize=(8, 5))
    plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')
    plt.axis('off')
    plt.title(f"Results: {DOMAIN}")
    plt.savefig(os.path.join(plot_dir, f'report_{DOMAIN}_{timestamp}.png'), bbox_inches='tight')
    plt.close()


save_report_as_image(y_true, y_pred, class_names, plot_dir, timestamp)

# Generate confusion matrices (raw counts and percentages)
cm = confusion_matrix(y_true, y_pred)

# Calculate percentage confusion matrix
row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_perc = np.divide(
    cm.astype('float'), row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0)

# Determine figure size based on number of classes
num_classes = len(class_names)
fig_size = max(10, num_classes * 0.8)

# Plot confusion matrix (percentages) - cleaner for reports
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm_perc * 100,  # Convert to percentages
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'Percentage (%)'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix: {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_{DOMAIN}_{timestamp}.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

# Plot confusion matrix (raw counts) - for detailed analysis
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix (Counts): {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_counts_{DOMAIN}_{timestamp}.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

# Generate Grad-CAM visualizations (RGB + Sobel branches).
explain_my_model(
    model=model,
    validation_data=val_dataset,
    save_dir=plot_dir,
    name_tag=timestamp,
    class_names=class_names,
)