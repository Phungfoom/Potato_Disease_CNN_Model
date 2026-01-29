import os
from tqdm import tqdm # progress bar for looks
import tensorflow as tf 

# Pre processing image files to be the same size format
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'hot_potato')
target_size = (224,224)
container_folders = {'field_classes', 'leaf_classes', 'tube_classes'}

print(f"'{input_dir}' and '{output_dir}' found with resolution '{target_size}'. Ignore this folder, '{container_folders}'")

image_files = []

print(f"'{input_dir}'Starting folder structure.")

for root, dir, files in os.walk(input_dir):
    data = os.path.basename(root)

    if data in container_folders:
        continue
    destination_folder = os.path.join(output_dir, data)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                full_path = os.path.join(root, file)
                image_files.append(full_path)

print(f"Folder structure done. {len(image_files)}")


print('Tensorflow: Resizing.')

for file_path in tqdm(image_files, desc = 'Resizing'):
    
    try:
        # Standarized RGB 
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_image(img_bytes, channels = 3, expand_animations = False)

        # Standarized image size (224x224)
        img_resized = tf.image.resize(img, target_size)

        # Save to new location
        disease_name = os.path.basename(os.path.dirname(file_path))
        filename = os.path.basename(file_path)
        new_path = os.path.join(output_dir, disease_name, filename)

        img_uint8 = tf.cast(img_resized, tf.uint8)
        encoded_img = tf.io.encode_jpeg(img_uint8, quality = 90)
        tf.io.write_file(new_path, encoded_img)

    except Exception as e:
        print(f"Skip corrupt files {file_path}: {e}")