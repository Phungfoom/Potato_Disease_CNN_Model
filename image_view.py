import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from build_sobel_model import sobel_edge_layer

#checking

def _resolve_stage_sources(base_dir: str, stage: str) -> list[tuple[str, str]]:
    """Return (label, directory) pairs to sample from for a pipeline stage."""
    stage = stage.strip().lower()
    if stage in ("train", "foundation", "lab"):
        stage = "training"
    if stage in ("field_test", "field-testing"):
        stage = "field"
    if stage == "training":
        sources = [("train", os.path.join(base_dir, "train"))]
        val_dir = os.path.join(base_dir, "val")
        if os.path.isdir(val_dir):
            sources.append(("val", val_dir))
        return sources
    if stage in ("field", "nasa"):
        return [("field_classes", os.path.join(base_dir, "field_classes"))]
    raise ValueError(f"Unsupported stage: {stage}")


def save_image_samples(stage, num_samples_per_class=2):
    """
    Generate and save RGB, grayscale, and Sobel visualizations for report.
    
    Saves images to outputs/image_samples/{stage}/ directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, config.BASE_DIR)
    
    # Set up output directory based on stage
    output_base = os.path.join(script_dir, config.OUTPUT_DIR, 'image_samples', stage)
    os.makedirs(output_base, exist_ok=True)
    
    sources = _resolve_stage_sources(base_dir, stage)

    for source_name, dataset_dir in sources:
        
        if not os.path.exists(dataset_dir):
            print(f'Folder not found: {dataset_dir}')
            continue
        
        # Use folder names as classes
        dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            shuffle=True,
            **config.DATA_PARAMS
        )
        
        class_names = dataset.class_names
        print(f"Sampling from '{source_name}': {len(class_names)} classes")
        if len(class_names) <= 1:
            print(
                f"Warning: source '{source_name}' has {len(class_names)} class. "
                "Outputs may look single-class only."
            )
        
        # Track how many samples we've collected per class
        class_counts = {class_name: 0 for class_name in class_names}
        total_needed = num_samples_per_class * len(class_names)
        
        # Collect samples from multiple batches to get variety
        for images, labels in dataset:
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                class_idx = int(labels[i].numpy())
                current_class = class_names[class_idx]
                
                # Skip if we already have enough samples for this class
                if class_counts[current_class] >= num_samples_per_class:
                    continue
                
                img_rgb = images[i] / 255.0
                img_gray = tf.image.rgb_to_grayscale(img_rgb)
                # 4d tensor for sobel
                img_sobel = sobel_edge_layer(img_gray[tf.newaxis, ...])
                img_sobel_plot = tf.squeeze(img_sobel).numpy()
                
                plt.figure(figsize=(15, 5))
                
                # RGB Image
                plt.subplot(1, 3, 1)
                plt.imshow(img_rgb)
                plt.title(f"RGB: {current_class}")
                plt.axis("off")
                
                # Grayscale Image
                plt.subplot(1, 3, 2)
                plt.imshow(img_gray.numpy().squeeze(), cmap='gray')
                plt.title(f"Grayscale: {current_class}")
                plt.axis("off")
                
                # Sobel Image
                plt.subplot(1, 3, 3)
                plt.imshow(img_sobel_plot, cmap='viridis')
                plt.title("Internal Feature:\nSobel Magnitude")
                plt.axis("off")
                
                plt.tight_layout()
                
                # Save with descriptive filename
                sample_num = class_counts[current_class] + 1
                filename = f"{source_name}_{current_class}_{sample_num}.png"
                save_path = os.path.join(output_base, filename)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                class_counts[current_class] += 1
                print(f"Saved: {save_path}")
            
            # Check if we've collected enough samples for all classes
            if all(count >= num_samples_per_class for count in class_counts.values()):
                break
        
        total_saved = sum(class_counts.values())
        print(
            f"Completed {source_name}: saved {total_saved} samples "
            f"across {len(class_names)} classes"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate image samples (RGB, grayscale, Sobel) for report."
    )
    parser.add_argument(
        '--stage',
        type=str,
        required=False,
        default='training',
        choices=['training', 'field', 'nasa', 'train', 'foundation', 'lab', 'field_test', 'field-testing'],
        help='Stage of the pipeline: training (after training/validation), '
             'field (field photos testing), or nasa (after NASA API)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=2,
        help='Number of samples to save per class (default: 2)'
    )
    args = parser.parse_args()
    
    save_image_samples(args.stage, args.num_samples)
    print(f"\nAll images saved to: outputs/image_samples/{args.stage}/")


if __name__ == "__main__":
    main()