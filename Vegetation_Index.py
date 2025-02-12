import os
import sys
import glob
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def list_subfolders(directory):
    """List all subfolders in the specified directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def find_band(directory, keyword):
    """Find a band file in a given directory based on a keyword."""
    for roots, dirs, files in os.walk(directory):
        search_pattern = os.path.join(roots, f"*{keyword}*.tif")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]
    return None

def load_band(tiff_path):
    """Load a band from a TIFF file and return its array and profile."""
    with rasterio.open(tiff_path) as src:
        band = src.read(1)
        profile = src.profile
        return band, profile

def resample_images(tiff_path, target_shape, target_profile):
    """Resample a TIFF image to match a target shape."""
    with rasterio.open(tiff_path) as src:
        target_profile.update({
            "width": target_shape[1],
            "height": target_shape[0],
            "transform": src.transform * src.transform.scale(
                (src.width / target_shape[1]),
                (src.height / target_shape[0])
            ) 
        })
        resampled_image = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear
        )
        return resampled_image, target_profile

def save_VI(output_path, band_paths, VI_name):
    """Generate a filename for the vegetation index output."""
    base_name = os.path.commonprefix([os.path.basename(band) for band in band_paths]).rstrip("_")
    output_filename = f"{base_name}_{VI_name}.tif"
    return os.path.join(output_path, output_filename)

def calculate_vi(band1, band2, band1_path, band2_path, profile, output_path, VI_name):
    """Calculate a vegetation index and save the results as an image and histogram."""
    if band2.shape != band1.shape:
        if band2.size > band1.size:
            band2, profile = resample_images(band2_path, band1.shape, profile) 
        else:
            band1, profile = resample_images(band1_path, band2.shape, profile)

    band1 = band1.astype("float32")
    band2 = band2.astype("float32")
    
    VI = (band1 - band2) / (band1 + band2 + 1e-6)
    
    # Save VI image
    VI_path = save_VI(output_path, [band2_path, band1_path], VI_name)
    plt.figure(figsize=(8,6))
    plt.imshow(VI, cmap="RdYlGn")
    plt.colorbar(label=VI_name)
    plt.title(f"{VI_name} Image")
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.savefig(VI_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Generate and save histogram
    histogram_path = VI_path.replace(f"_{VI_name}.tif", f"_{VI_name}_Histogram.png")
    plt.figure(figsize=(8,6))
    plt.hist(VI.flatten(), bins=50, color="purple", alpha=0.75)
    plt.title(f"{VI_name}: Distribution of pixels")
    plt.xlabel(VI_name)
    plt.ylabel("Frequency")
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()

    return VI

def process_folder(main_directory, subfolder, output_path):
    """Process a subfolder, calculating vegetation indices and generating images."""
    band_folder = os.path.join(main_directory, subfolder)
    green_band_path = find_band(band_folder, "Green")
    red_band_path = find_band(band_folder, "Red")
    red_edge_band_path = find_band(band_folder, "RedEdge")
    nir_band_path = find_band(band_folder, "NIR")
    
    if not green_band_path or not red_band_path or not red_edge_band_path or not nir_band_path:
        print(f"Skipping {subfolder} due to missing bands.")
        return
    
    green_band, profile = load_band(green_band_path)
    red_band, _ = load_band(red_band_path)
    red_edge_band, _ = load_band(red_edge_band_path)
    nir_band, _ = load_band(nir_band_path)

    calculate_vi(nir_band, red_band, nir_band_path, red_band_path, profile, output_path, "NDVI")
    calculate_vi(nir_band, red_edge_band, nir_band_path, red_edge_band_path, profile, output_path, "NDRE")
    calculate_vi(nir_band, green_band, nir_band_path, green_band_path, profile, output_path, "GNDVI")

def main():
    """Main function to prompt user and process vegetation indices."""
    main_directory = "Put_Your_Main_Directory"
    output_path = "Put_Your_Output_Path"
    
    subfolders = list_subfolders(main_directory)
    
    if not subfolders:
        print("No subfolders found in 'Vegetation_Index'.")
        sys.exit(1)
    
    print("Available subfolders:")
    for i, subfolder in enumerate(subfolders, start=1):
        print(f"{i}. {subfolder}")

    choice = input("Enter the number of the subfolder you want to use or type 'all' to process all folders: ")
    
    if choice.lower() == "all":
        for subfolder in subfolders:
            process_folder(main_directory, subfolder, output_path)
    else:
        try:
            choice = int(choice)
            if 1 <= choice <= len(subfolders):
                selected_subfolder = subfolders[choice - 1]
                process_folder(main_directory, selected_subfolder, output_path)
            else:
                raise ValueError
        except ValueError:
            print("Invalid selection. Please run the script again and enter a valid number.")
            sys.exit(1)

if __name__ == "__main__":
    main()
