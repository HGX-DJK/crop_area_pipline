import pandas as pd
import numpy as np

csv_path = "crop_area_pipeline/data/sample_training_points.csv"
df = pd.read_csv(csv_path)

# Check if already added
if not any(df["crop_name"] == "亚热带常绿山地森林"):
    np.random.seed(42)
    n_new = 400
    start_id = len(df) + 1
    new_rows = []
    for i in range(n_new):
        pid = f"P{start_id + i:04d}"
        d3 = round(float(np.random.uniform(0.64, 0.72)), 4)
        d6 = round(float(np.random.uniform(0.83, 0.91)), 4)
        d9 = round(float(np.random.uniform(0.74, 0.83)), 4)
        new_rows.append({
            "point_id": pid,
            "label": 0,
            "crop_name": "亚热带常绿山地森林",
            "doy_3": d3,
            "doy_6": d6,
            "doy_9": d9
        })
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"成功向 {csv_path} 追加 {n_new} 条高精度常绿山地森林负样本！当前总样本数: {len(df_combined)}")
else:
    print(f"{csv_path} 中已存在常绿山地森林样本，跳过追加。当前总样本数: {len(df)}")
