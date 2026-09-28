import numpy as np

# Let's inspect what happens in predict_raster_cube
classes_arr = np.array([0, 1])
probs = np.array([[0.95, 0.05], [0.10, 0.90]]) # pixel 0 is bg (95%), pixel 1 is crop (90%)

# What old code did:
max_probs = np.max(probs, axis=1) # [0.95, 0.90]
# What main.py did:
crop_mask = (max_probs >= 0.50) # [True, True] -> BOTH BECAME CROPLAND!

print("Old code max_probs:", max_probs)
print("Old code crop_mask (both became crop!):", crop_mask)

# Correct crop probability:
# If binary: crop_prob is prob of class 1!
crop_prob = probs[:, 1]
print("Correct crop_prob:", crop_prob)
print("Correct crop_mask:", crop_prob >= 0.50)
