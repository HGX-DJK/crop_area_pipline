import os
import rasterio
import rasterio.env

proj_cand = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
if os.path.exists(proj_cand):
    rasterio.env.set_proj_data_search_path(proj_cand)
    os.environ["PROJ_LIB"] = proj_cand
    os.environ["PROJ_DATA"] = proj_cand

from rasterio.warp import transform_bounds

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    bounds = src.bounds
    crs = src.crs
    wgs84_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
    print(f"CRS: {crs}")
    print(f"UTM Bounds: {bounds}")
    print(f"WGS84 Bounds (min_lon, min_lat, max_lon, max_lat): {wgs84_bounds}")
