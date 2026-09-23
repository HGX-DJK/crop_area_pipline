import json
import rasterio
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape

print("=== 1. 验证贝里埃萨湖/海岸山脉高山林木压制效果 ===")
with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    x, y = transformer.transform(-122.08, 38.65)
    r, c = src.index(x, y)
    patch = mask[r-50:r+50, c-50:c+50]
    crop_pct = np.mean(patch == 1) * 100.0
    print(f"(-122.08, 38.65) Coast Range 100x100 窗口耕地率: {crop_pct:.2f}% (此前为 99.9%)")
    assert crop_pct <= 0.1, f"Expected <= 0.1%, got {crop_pct}%"

    # Also check Lake Berryessa center (-122.23, 38.58)
    x_lake, y_lake = transformer.transform(-122.23, 38.58)
    r_l, c_l = src.index(x_lake, y_lake)
    patch_lake = mask[r_l-30:r_l+30, c_l-30:c_l+30]
    print(f"(-122.23, 38.58) Lake Berryessa 60x60 窗口耕地率: {np.mean(patch_lake == 1)*100.0:.2f}%")

    # Plain farmland around Davis (-121.75, 38.54)
    x_davis, y_davis = transformer.transform(-121.75, 38.54)
    r_d, c_d = src.index(x_davis, y_davis)
    patch_davis = mask[r_d-30:r_d+30, c_d-30:c_d+30]
    print(f"(-121.75, 38.54) Davis Plain Farmland 60x60 窗口耕地率: {np.mean(patch_davis == 1)*100.0:.2f}%")

print("\n=== 2. 验证矢量地块形状指标与长宽比 (杜绝条状伪影) ===")
with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

features = geojson_data['features']
print(f"导出的主力地块数量: {len(features)}")

aspect_ratios = []
compactness_list = []
areas_mu = []

for feat in features:
    props = feat['properties']
    geom = shape(feat['geometry'])
    minx, miny, maxx, maxy = geom.bounds
    # Estimate width and height in meters (approx at lat 38.5)
    w_m = (maxx - minx) * 111320.0 * np.cos(np.radians(38.5))
    h_m = (maxy - miny) * 110574.0
    aspect = max(w_m, h_m) / max(min(w_m, h_m), 1.0)
    aspect_ratios.append(aspect)
    compactness_list.append(props.get('compactness', 0.0))
    areas_mu.append(props.get('area_mu', 0.0))

print(f"地块平均面积: {np.mean(areas_mu):.1f} 亩 (中位数: {np.median(areas_mu):.1f} 亩)")
print(f"地块长宽比统计:")
print(f"  平均长宽比: {np.mean(aspect_ratios):.2f}")
print(f"  中位数长宽比: {np.median(aspect_ratios):.2f}")
print(f"  长宽比 <= 2.0 的地块占比: {np.mean(np.array(aspect_ratios) <= 2.0)*100:.1f}%")
print(f"  长宽比 <= 3.0 的地块占比: {np.mean(np.array(aspect_ratios) <= 3.0)*100:.1f}%")
print(f"  极端条状地块 (长宽比 > 5.0) 数量: {np.sum(np.array(aspect_ratios) > 5.0)}")

print("\n=== 3. 抽样查看前 10 大主力地块形态 ===")
for i in range(10):
    p = features[i]['properties']
    print(f"  {p['parcel_id']}: 面积={p['area_mu']} 亩, 紧凑度={p['compactness']}, 长宽比={aspect_ratios[i]:.2f}, 中心=({p['center_lon']:.4f}, {p['center_lat']:.4f})")
