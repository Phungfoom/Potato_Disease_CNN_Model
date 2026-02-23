import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from build_rgb_model import build_rgb_model
from build_grayscale_model import build_grayscale_model

# Terminal commands: different commands for tube, lead and field cases 

parser = argparse.ArgumentParser(description ="Train Potato ID Models")
parser.add_argument('--domain', type = str, required = True, choices = ['leaf', 'tube', 'field'], 
                    help = "Which dataset to train on: 'leaf', 'tube', or 'field'")
parser.add_argument('--mode', type=str, required = True, choices = ['rgb', 'grayscale'], 
                    help = "Which model to use: 'rgb' or 'grayscale'")

args = parser.parse_args()

DOMAIN = args.domain
TRAINING_MODE = args.mode.upper()

# Grayscale or RGB
if TRAINING_MODE == "GRAYSCALE":
    c_mode = "grayscale"
    in_shape = (224, 224, 1)
else:
    c_mode = "rgb"
    in_shape = (224, 224, 3)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Folder for both models
model_dir = os.path.join(script_dir, 'outputs', TRAINING_MODE.lower(), 'models')
plot_dir = os.path.join(script_dir, 'outputs', TRAINING_MODE.lower(), 'plots')

os.makedirs(model_dir, exist_ok = True)
os.makedirs(plot_dir, exist_ok = True)

# Save file for each model and class
model_save_path = os.path.join(model_dir, f'potato_{DOMAIN}_{c_mode}_model.keras')
plot_save_path = os.path.join(plot_dir, f'potato_{DOMAIN}_{c_mode}_history.png')

# Sub folders
dataset_dir = os.path.join(script_dir, 'hot_potato', f'{DOMAIN}_classes')

batch_size = 32
img_size = (224, 224)
seed = 123

print(f"loading data {dataset_dir}")

# Training data (70%)
train_spud = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.3,
    subset = "training",
    seed = 123,
    image_size = img_size,
    batch_size = batch_size,
    color_mode = c_mode,
    shuffle = True  
)

# Validation data (30%)
val_spud = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.3,
    subset = 'validation',
    seed = 123,
    image_size = img_size,
    batch_size = batch_size,
    color_mode = c_mode,
    shuffle = False
)

class_names = train_spud.class_names
num_classes = len(class_names)

print(f"Number of classes {num_classes}, classes: {class_names}")

# Normalize 

AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1./255) # black, white, gray

def preprocess(image, label):
    return normalization_layer(image), label

train_spud = train_spud.map(preprocess).cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_spud = val_spud.map(preprocess).cache().prefetch(buffer_size = AUTOTUNE)

# Model
 
if TRAINING_MODE == "GRAYSCALE":
    model = build_grayscale_model(input_shape=in_shape, num_classes=num_classes)
else:
    model = build_rgb_model(input_shape=in_shape, num_classes=num_classes)

EPOCHS = 10

history = model.fit(
    train_spud,
    validation_data=val_spud,
    epochs = EPOCHS,
    verbose = 1
)

print(f'Saving training plot to {plot_save_path}...')
model.save(model_save_path)
print('model saved')

# Visualize 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
plt.figure(figsize = (12, 6))

# Plot: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title(f'{DOMAIN.capitalize()} ({TRAINING_MODE}) - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title(f'{DOMAIN.capitalize()} ({TRAINING_MODE}) - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

print(f'Saving training plot to {plot_save_path}...')
plt.savefig(plot_save_path)

plt.show()







