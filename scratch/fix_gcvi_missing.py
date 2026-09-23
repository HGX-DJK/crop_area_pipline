import re

with open('src/raster_loader.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "lswi[valid_lswi] = (b4[valid_lswi] - b5[valid_lswi]) / denom_lswi[valid_lswi]" in line:
        new_lines.append("""
    # --- GCVI (Green Chlorophyll Vegetation Index) ---
    gcvi = np.zeros_like(b2)
    valid_gcvi = b2 > 0
    gcvi[valid_gcvi] = (b4[valid_gcvi] / b2[valid_gcvi]) - 1.0
    gcvi = np.clip(gcvi, -2.0, 10.0)
""")

with open('src/raster_loader.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Added gcvi calculation to raster_loader.py")
