import tensorflow as tf
import matplotlib.pyplot as plt
import os

dataset_dir = os.path.join('hot_potato', 'leaf_classes')

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(224, 224),
    batch_size=1,
    shuffle=True # Shuffles so you get a random image every time!
)

for images, labels in dataset.take(1):
    img_rgb = images[0] / 255.0  
    
    img_gray = tf.image.rgb_to_grayscale(img_rgb)

    plt.figure(figsize = (10, 5))

    # Plot RGB Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("RGB Input")
    plt.axis("off")

    # Plot Grayscale Image
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray.numpy().squeeze(), cmap='gray')
    plt.title("Grayscale Input")
    plt.axis("off")

    plt.show()