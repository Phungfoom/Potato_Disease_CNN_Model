import os
import tensorflow as tf
import matplotlib.pyplot as plt


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

# Normalize 

AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1./255) # black, white, gray

def preprocess(image, label):
    return normalization_layer(image), label

train_spud = train_spud.map(preprocess).cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_spud = val_spud.map(preprocess).cache().prefetch(buffer_size = AUTOTUNE)

# Model
 
from build_rgb_model import build_rgb_model
model = build_rgb_model(input_shape = (224, 224, 3), num_classes = num_classes)

EPOCHS = 10

history = model.fit(
    train_spud,
    validation_data=val_spud,
    epochs = EPOCHS,
    verbose=1
)

print('saving model...')
model.save('potato_rgb_model.keras')
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
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True)

# Plot: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.grid(True)

plt.show()







