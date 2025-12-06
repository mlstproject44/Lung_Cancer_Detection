import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os

def downsample_scan(input_path, output_path, target_spacing=(2.0, 2.0, 2.5)):
    """
    Downsample a CT scan to reduce file size

    Parameters:
    -----------
    input_path : str
        Path to input .mhd file
    output_path : str
        Path to save downsampled .mhd file
    target_spacing : tuple
        Desired voxel spacing in mm (x, y, z)
        Examples:
        - (1.0, 1.0, 2.5) = mild downsampling (~70% size reduction)
        - (2.0, 2.0, 2.5) = moderate (~90% size reduction) ← recommended
        - (3.0, 3.0, 3.0) = aggressive (~98% size reduction)

    Returns:
    --------
    Size reduction percentage
    """
    print(f"Processing: {os.path.basename(input_path)}")

    scan = sitk.ReadImage(input_path)

    original_spacing = scan.GetSpacing()
    original_size = scan.GetSize()

    print(f"  Original spacing: {original_spacing}")
    print(f"  Original size: {original_size}")

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    print(f"  Target spacing: {target_spacing}")
    print(f"  New size: {new_size}")

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(scan.GetDirection())
    resample.SetOutputOrigin(scan.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(scan.GetPixelIDValue())

    # Use linear interpolation (fast and good enough for CT)
    resample.SetInterpolator(sitk.sitkLinear)

    # Perform resampling
    downsampled_scan = resample.Execute(scan)

    # Save
    sitk.WriteImage(downsampled_scan, output_path)

    # Calculate size reduction
    original_bytes = np.prod(original_size) * 2  # int16 = 2 bytes
    new_bytes = np.prod(new_size) * 2
    reduction = (1 - new_bytes / original_bytes) * 100

    original_mb = original_bytes / (1024**2)
    new_mb = new_bytes / (1024**2)

    try:
        print(f"  Size: {original_mb:.1f} MB -> {new_mb:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Saved to: {os.path.basename(output_path)}\n")
    except:
        # Fallback for encoding issues
        print(f"  Size: {original_mb:.1f} MB to {new_mb:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Saved!\n")

    return reduction

def batch_downsample(input_dir, output_dir, target_spacing=(2.0, 2.0, 2.5)):
    """
    Downsample all CT scans in a directory

    Parameters:
    -----------
    input_dir : str
        Directory containing .mhd files
    output_dir : str
        Directory to save downsampled scans
    target_spacing : tuple
        Target voxel spacing in mm
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all .mhd files
    input_path = Path(input_dir)
    mhd_files = list(input_path.glob('*.mhd'))

    if len(mhd_files) == 0:
        print(f"No .mhd files found in {input_dir}")
        return

    print(f"Found {len(mhd_files)} scans to process\n")
    print("="*70)

    total_original_size = 0
    total_new_size = 0

    for i, mhd_file in enumerate(mhd_files, 1):
        print(f"[{i}/{len(mhd_files)}]")

        # Input and output paths
        input_path = str(mhd_file)
        output_filename = mhd_file.name
        output_path = os.path.join(output_dir, output_filename)

        # Downsample
        try:
            reduction = downsample_scan(input_path, output_path, target_spacing)

            # Track sizes
            scan = sitk.ReadImage(input_path)
            original_size = np.prod(scan.GetSize()) * 2
            total_original_size += original_size
            total_new_size += original_size * (1 - reduction/100)

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Processed: {len(mhd_files)} scans")
    if total_original_size > 0:
        print(f"Original total size: {total_original_size/(1024**3):.2f} GB")
        print(f"New total size: {total_new_size/(1024**3):.2f} GB")
        print(f"Total space saved: {(total_original_size - total_new_size)/(1024**3):.2f} GB")
        print(f"Overall reduction: {(1 - total_new_size/total_original_size)*100:.1f}%")
    else:
        print("No scans were successfully processed")

def compare_quality(original_path, downsampled_path, slice_idx=None):
    """
    Compare original and downsampled scans visually
    """
    import matplotlib.pyplot as plt

    # Load both scans
    original = sitk.ReadImage(original_path)
    downsampled = sitk.ReadImage(downsampled_path)

    original_array = sitk.GetArrayFromImage(original)
    downsampled_array = sitk.GetArrayFromImage(downsampled)

    # Use middle slice if not specified
    if slice_idx is None:
        slice_idx = original_array.shape[0] // 2

    # Find corresponding slice in downsampled
    scale_factor = original_array.shape[0] / downsampled_array.shape[0]
    downsampled_slice_idx = int(slice_idx / scale_factor)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(original_array[slice_idx], cmap='gray')
    axes[0].set_title(f'Original - {original.GetSize()}\nSpacing: {original.GetSpacing()}')
    axes[0].axis('off')

    axes[1].imshow(downsampled_array[downsampled_slice_idx], cmap='gray')
    axes[1].set_title(f'Downsampled - {downsampled.GetSize()}\nSpacing: {downsampled.GetSpacing()}')
    axes[1].axis('off')

    plt.tight_layout()
    output_path = original_path.replace('.mhd', '_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_path}")
    plt.close()

# =============================================================================
# MAIN - Examples
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CT SCAN DOWNSAMPLING")
    print("="*70)
    print()

    # Configuration
    input_dir = r"C:\Users\User\Downloads\subset8\subset8"
    output_dir = r"C:\Users\User\Desktop\downsampled_subset8"

    # Choose your target spacing:
    # - (1.0, 1.0, 2.5) = Mild downsampling, high quality (~30% reduction)
    # - (2.0, 2.0, 2.5) = Moderate, good balance (~90% reduction) ← RECOMMENDED
    # - (3.0, 3.0, 3.0) = Aggressive, very small (~98% reduction)
    target_spacing = (2.0, 2.0, 2.5)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target spacing: {target_spacing} mm\n")

    # Process all scans
    batch_downsample(input_dir, output_dir, target_spacing)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nDownsampled scans saved to: {output_dir}")
    print("\nTo change resolution, edit the 'target_spacing' variable:")
    print("  - Smaller values = higher quality, larger files")
    print("  - Larger values = lower quality, smaller files")
