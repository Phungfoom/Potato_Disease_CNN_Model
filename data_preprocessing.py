import os
from tqdm import tqdm # progress bar for looks
import tensorflow as tf 

# Pre processing image files to be the same size format
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'hot_potato')
target_size = (224,224)

image_files = []

print(f"'{input_dir}'Starting folder structure.")

for root, dir, files in os.walk(input_dir):
    relative_path = os.path.relpath(root, input_dir)

    if relative_path == '.':
        continue
    
    destination_folder = os.path.join(output_dir, relative_path)
    os.makedirs(destination_folder, exist_ok = True)

    for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                full_input_path = os.path.join(root, file)
                full_output_path = os.path.join(destination_folder, file)
            
                image_files.append((full_input_path, full_output_path))


print('Tensorflow: Resizing.')

for input_path, output_path in tqdm(image_files, desc='Resizing'):
    try:
        # Standardized RGB 
        img_bytes = tf.io.read_file(input_path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

        # Standardized image size (224x224)
        img_resized = tf.image.resize(img, target_size, method = 'pad', antialias = True)

        # Convert and save directly to the perfectly mapped output_path
        img_uint8 = tf.cast(img_resized, tf.uint8)
        encoded_img = tf.io.encode_jpeg(img_uint8, quality = 90)
        tf.io.write_file(output_path, encoded_img)

    except Exception as e:
        print(f"Skip corrupt file {input_path}: {e}")