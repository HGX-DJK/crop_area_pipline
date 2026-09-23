import rasterio
import numpy as np
import cv2

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    win = rasterio.windows.Window(3000, 1500, 500, 500)
    b4 = src.read(4, window=win).astype(np.float32)
    b3 = src.read(3, window=win).astype(np.float32)
    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    
    # Calculate gradient / Sobel on NDVI
    grad_x = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    print(f"NDVI gradient magnitude in 500x500 farmland patch:")
    print(f"  Min={np.min(grad_mag):.4f}, Mean={np.mean(grad_mag):.4f}, Max={np.max(grad_mag):.4f}")
    print(f"  75th percentile={np.percentile(grad_mag, 75):.4f}, 90th percentile={np.percentile(grad_mag, 90):.4f}")
    
    # Notice: field boundaries have high gradient (grad_mag > 0.08)
    strong_edges = grad_mag > 0.08
    print(f"Strong boundary edges percentage: {np.mean(strong_edges)*100:.1f}%")
