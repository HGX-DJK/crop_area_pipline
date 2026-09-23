import os
import rasterio
import rasterio.env

proj_cand = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
if os.path.exists(proj_cand):
    rasterio.env.set_proj_data_search_path(proj_cand)
    os.environ["PROJ_LIB"] = proj_cand
    os.environ["PROJ_DATA"] = proj_cand

from rasterio.warp import transform

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    # Let's convert lon -122.0, lat 38.5 to pixel row, col
    xs, ys = transform('EPSG:4326', src.crs, [-122.05, -121.95, -121.85], [38.45, 38.45, 38.45])
    for lon, x, y in zip([-122.05, -121.95, -121.85], xs, ys):
        row, col = src.index(x, y)
        print(f"Lon {lon}: UTM X={x:.1f}, Y={y:.1f} -> Row={row}, Col={col}")
