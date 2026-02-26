import config
from build_rgb_model import build_rgb_model
from build_grayscale_model import build_grayscale_model
from build_sobel_model import build_sobel_model

print(f"Config Image Size: {config.DATA_PARAMS['image_size']}")

try:
    m1 = build_rgb_model()
    print("✅ RGB Branch: OK (Input:", m1.input_shape, ")")
    
    m2 = build_grayscale_model()
    print("✅ Gray Branch: OK (Input:", m2.input_shape, ")")
    
    m3 = build_sobel_model()
    print("✅ Sobel Branch: OK (Input:", m3.input_shape, ")")
    
    print("\n All branches are synced with config.py")
except Exception as e:
    print(f"\n Sync Error: {e}")