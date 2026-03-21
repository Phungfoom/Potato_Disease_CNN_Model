# Data dimensions
DATA_PARAMS = {
    "image_size": (224, 224),
    "batch_size": 32,
    "seed": 123}

DROPOUT_RATE = 0.3
EPOCHS = 10

# Paths
# Folder that holds the processed training data (e.g. leaf_classes, tube_classes, etc.).
# All scripts that load images should build paths from this base.
BASE_DIR = 'processed_data'
OUTPUT_DIR = 'outputs'

# Bump this when you change preprocessing (_to_triple, Sobel path, normalization, etc.) so
# tf.data file caches are not reused incorrectly. Image size / batch / --augment are
# already encoded in the cache basename in train_model.py.
TF_CACHE_PIPELINE_ID = "v1"