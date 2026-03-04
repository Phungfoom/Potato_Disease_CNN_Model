import tensorflow as tf
import matplotlib.pyplot as plt
import os
import config

from build_sobel_model import sobel_edge_layer

# Random images for report
base_dir = 'hot_potato'
categories = ['leaf_classes','tube_classes']

# Loop to go through folder categories
for cat in categories:
    dataset_dir = os.path.join(base_dir, cat)

    if not os.path.exists(dataset_dir):
        print('Folder not found')
        continue 

    # Use folder names as classes
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        shuffle = True,
        **config.DATA_PARAMS)

    class_names = dataset.class_names

    for images, labels in dataset.take(1):
        img_rgb = images[0] / 255.0  
        img_gray = tf.image.rgb_to_grayscale(img_rgb)
        # 4d tensor for sobel
        img_sobel = sobel_edge_layer(img_gray[tf.newaxis, ...])
        img_sobel_plot = tf.squeeze(img_sobel).numpy()

        current_class = class_names[labels[0]]
    
        plt.figure(figsize = (15, 5))

        # RGB Image
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title(f"RGB: {current_class}")
        plt.axis("off")

        # Grayscale Image
        plt.subplot(1, 3, 2)
        plt.imshow(img_gray.numpy().squeeze(), cmap = 'gray')
        plt.title(f"Grayscale: {current_class}")
        plt.axis("off")

        # Sobel Image
        plt.subplot(1, 3, 3)
        plt.imshow(img_sobel_plot, cmap='viridis') 
        plt.title("Internal Feature:\nSobel Magnitude")
        plt.axis("off")

        plt.show()