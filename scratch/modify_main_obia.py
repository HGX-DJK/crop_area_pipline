import re

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "logger.info(\"[" in line and "3/5]" in line:
        new_lines.append('    logger.info("[步骤 3/5] 执行 OBIA 面向对象超像素分割 (SLIC)...")\n')
        new_lines.append('    edge_mask = None\n')
        new_lines.append('    optical_rgb = None\n')
        new_lines.append('    if input_mode == "geotiff" and sorted_files:\n')
        new_lines.append('        edge_mask = loader.compute_spectral_edge_mask(sorted_files[0], crop_mask)\n')
        new_lines.append('        import rasterio\n')
        new_lines.append('        import numpy as np\n')
        new_lines.append('        try:\n')
        new_lines.append('            with rasterio.open(sorted_files[-1]) as src:\n')
        new_lines.append('                b3 = src.read(3).astype(np.float32)\n')
        new_lines.append('                b2 = src.read(2).astype(np.float32)\n')
        new_lines.append('                b1 = src.read(1).astype(np.float32)\n')
        new_lines.append('                def stretch(band):\n')
        new_lines.append('                    p2, p98 = np.percentile(band, (2, 98))\n')
        new_lines.append('                    stretched = np.clip((band - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255)\n')
        new_lines.append('                    return stretched.astype(np.uint8)\n')
        new_lines.append('                optical_rgb = np.dstack([stretch(b3), stretch(b2), stretch(b1)])\n')
        new_lines.append('                logger.info(f"  -> 成功加载真彩色光学底图用于 SLIC 指导，形状: {optical_rgb.shape}")\n')
        new_lines.append('        except Exception as e:\n')
        new_lines.append('            logger.warning(f"无法加载光学底图供SLIC使用: {e}")\n')
        new_lines.append('    segmenter = ParcelSegmenter(config)\n')
        new_lines.append('    parcel_id_mask, parcel_metadata = segmenter.segment_parcels(crop_mask, conf_map, edge_mask=edge_mask, optical_image=optical_rgb)\n')
        skip = True
    elif skip and "total_valid_parcels = len(parcel_metadata)" in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Updated main.py")
