import rasterio, cv2, time, numpy as np

t0 = time.time()
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
t1 = time.time()
gx = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
mag = np.sqrt(gx**2 + gy**2)
t2 = time.time()
print(f"Read bands: {t1-t0:.2f}s, Sobel gradient on 3660x3660: {t2-t1:.2f}s")
