import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Random images for report
base_dir = 'hot_potato'
categories = ['leaf_classes', 'field_classes', 'tube_classes']

# Loop to go through folder categories
for cat in categories:
    dataset_dir = os.path.join(base_dir, cat)

    if not os.path.exists(dataset_dir):
        print('Folder not found')
        continue 

    # Use folder names as classes
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size = (224, 224),
        batch_size = 1,
        shuffle = True)

    class_names = dataset.class_names

    for images, labels in dataset.take(1):
        img_rgb = images[0] / 255.0  
        img_gray = tf.image.rgb_to_grayscale(img_rgb)

        current_class = class_names[labels[0]]
    
        plt.figure(figsize = (10, 5))

        # RGB Image
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"RGB: {current_class}")
        plt.axis("off")

        # Grayscale Image
        plt.subplot(1, 2, 2)
        plt.imshow(img_gray.numpy().squeeze(), cmap = 'gray')
        plt.title(f"Grayscale: {current_class}")
        plt.axis("off")

        plt.show()