import os
import argparse
import tensorflow as tf
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm
from Potato_bunch_brain import build_combined_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Focus on leaf and tubr for foundation, field is midterm
parser = argparse.ArgumentParser(description = "Train Potato ID Fusion Model")
parser.add_argument('--domain', type = str, required = True, choices = ['leaf', 'tube', 'field'])
parser.add_argument('--stage', type = str, default = "foundation", choices = ['foundation', 'midterm'])
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = args.domain
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Train on foundation, then midterm for fields, and NASA API for final check

if args.stage == "foundation":
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'{DOMAIN}_classes')
else:
    # Example: Midterm might use a different directory for "Field" tests
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'midterm_{DOMAIN}_data')

train_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset="training",
    color_mode = "rgb",
    shuffle = True,
    **config.DATA_PARAMS)

# weights for abundance of some photos over another 
y_train = np.concatenate([y for _, y in train_raw], axis = 0)

weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train),
    y = y_train)
class_weights_dict = dict(enumerate(weights))

class_names = train_raw.class_names
num_classes = len(class_names)

val_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = "validation",
    color_mode = "rgb",
    shuffle = True,
    **config.DATA_PARAMS)

val_labels = np.concatenate([y for _, y in val_raw], axis=0)
print("Validation images per class:", dict(zip(class_names, np.bincount(val_labels))))

# Inputs for 3 branches 
def prepare_triple_input(image, label):
    norm_image = tf.cast(image, tf.float32) / 255.0 # Normalize 
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    sobel_image = tf.image.sobel_edges(gray_image)
    sobel_image = tf.reduce_sum(tf.abs(sobel_image), axis = -1)

    return (norm_image, gray_image, sobel_image), label # rgb, gray, sobel 

train_spud = train_raw.map(prepare_triple_input).cache('spud_cache').prefetch(tf.data.AUTOTUNE)
val_spud = val_raw.map(prepare_triple_input).cache('spud_cache').prefetch(tf.data.AUTOTUNE)

# training history 
def plot_training_history(history, plot_dir, domain, stage, timestamp):
    sns.set_theme(style = "whitegrid")
    plt.subplots(1, 2, figsize = (15, 5))

    # Accuracy
    sns.lineplot(x = range(len(history.history['accuracy'])), 
                 y = history.history['accuracy'], label = 'Train Acc')
    sns.lineplot(x=range(len(history.history['val_accuracy'])), 
                 y = history.history['val_accuracy'], label = 'Val Acc')
    plt.title(f'Accuracy: {domain} ({stage})')

    # Loss
    sns.lineplot(x = range(len(history.history['loss'])), 
                 y = history.history['loss'], label = 'Train Loss')
    sns.lineplot(x = range(len(history.history['val_loss'])), 
                 y = history.history['val_loss'], label = 'Val Loss')
    plt.title(f'Loss: {domain} ({stage})')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'history_{domain}_{stage}_{timestamp}.png'))    
    plt.close()

model = build_combined_model(num_classes = num_classes)

# Folders for outputs
model_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'models')
plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'plots')
os.makedirs(model_dir, exist_ok = True)
os.makedirs(plot_dir, exist_ok = True)

history = model.fit(
    train_spud,
    validation_data = val_spud,
    epochs = config.EPOCHS,
    class_weight = class_weights_dict)

model.save(os.path.join(model_dir, f'potato_{DOMAIN}_model_{timestamp}.keras'))
plot_training_history(history, plot_dir, DOMAIN, args.stage, timestamp)

# Confusion Matrtix for predictions
y_true, y_pred = [], []

# Classification Report
for images, labels in tqdm(val_spud, desc = "Evaluating"):   
    preds = model.predict_on_batch(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis = 1))

def save_report_as_image(y_true, y_pred, target_names, plot_dir):
    # Create the report as a dictionary
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    plt.figure(figsize=(8, 4))
    plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')
    plt.axis('off')
    plt.title(f"Results: {DOMAIN}")
    plt.savefig(os.path.join(plot_dir, f'report_{DOMAIN}_{timestamp}.png'), bbox_inches = 'tight')
    plt.close()

save_report_as_image(y_true, y_pred, class_names, plot_dir)

cm = confusion_matrix(y_true, y_pred)

row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_perc = np.divide(cm.astype('float'), row_sums, 
                    out=np.zeros_like(cm, dtype=float), 
                    where=row_sums != 0)

plt.figure(figsize = (10, 8))
sns.heatmap(cm, annot = True, fmt = 'd', 
            xticklabels = class_names, 
            yticklabels = class_names)
plt.title(f'Confusion Matrix: {DOMAIN} Fusion Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(plot_dir, f'cm_{DOMAIN}_{timestamp}.png'), bbox_inches = 'tight')
plt.close()

plt.figure(figsize = (12, 10)) 
sns.heatmap(cm_perc, annot = True, fmt = '.2f', xticklabels = class_names, yticklabels=class_names, cmap = 'Blues')
plt.title(f'Confusion Matrix (Percentages): {DOMAIN}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(plot_dir, f'cm_percent_{DOMAIN}_{timestamp}.png'), bbox_inches = 'tight')
plt.close()