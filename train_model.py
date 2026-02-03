import os
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, 'hot_potato')

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
    image_size = (224, 224),
    batch_size = 32,
    shuffle = True  
)

# Validation data (30%)
val_spud = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.3,
    subset = 'validation',
    seed = 123,
    image_size = (224, 224),
    batch_size = 32,
    shuffle = False
)

class_names = train_spud.class_names
num_classes = len(class_names)

print(f"Number of classes {num_classes}, classes: {class_names}")