import re

with open('src/raster_loader.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'is_mountain_veg = is_rough_mountain | is_dense_forest'
replacement = '''is_shrub_or_tea = (ndvi > 0.45) & (cv_b4 > 0.035) & (b3 < 1000.0) & (b5 < 2400.0)
    is_mountain_veg = is_rough_mountain | is_dense_forest | is_shrub_or_tea'''

if target in content:
    content = content.replace(target, replacement)
    with open('src/raster_loader.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully modified raster_loader.py')
else:
    print('Target string not found')
