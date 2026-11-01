# Grad-CAM++ checks (main, val, n=274)

Median [IQR] per case. *_mass = share of heatmap mass; *_vol = share of brain volume; *_ratio = mass / volume (1 = no enrichment). shell = outer 3 voxels (6 mm) of the brain. drop_* = fall in tumour probability after zeroing the top-k% CAM (or random) brain voxels.

## tumour

| metric | median | IQR | n |
|---|---|---|---|
| det_drop_cam_10 | 0.000 | 0.000 to 0.000 | 188 |
| det_drop_cam_20 | 0.000 | 0.000 to 0.000 | 188 |
| det_drop_cam_30 | 0.000 | 0.000 to 0.000 | 188 |
| det_drop_random_10 | 0.000 | -0.000 to 0.000 | 188 |
| det_drop_random_20 | 0.000 | -0.000 to 0.000 | 188 |
| det_drop_random_30 | 0.000 | -0.000 to 0.000 | 188 |
| det_outside_mass | 0.702 | 0.681 to 0.723 | 188 |
| det_shell_mass | 0.058 | 0.055 to 0.060 | 188 |
| det_shell_ratio | 0.244 | 0.232 to 0.261 | 188 |
| det_shell_vol | 0.235 | 0.229 to 0.241 | 188 |
| det_tumour_mass | 0.021 | 0.011 to 0.031 | 188 |
| det_tumour_ratio | 0.325 | 0.257 to 0.398 | 188 |
| det_tumour_vol | 0.060 | 0.034 to 0.099 | 188 |
| seg_outside_mass | 0.221 | 0.148 to 0.389 | 188 |
| seg_shell_mass | 0.078 | 0.067 to 0.098 | 188 |
| seg_shell_ratio | 0.332 | 0.281 to 0.420 | 188 |
| seg_shell_vol | 0.235 | 0.229 to 0.241 | 188 |
| seg_tumour_mass | 0.555 | 0.349 to 0.646 | 188 |
| seg_tumour_ratio | 7.642 | 6.046 to 11.269 | 188 |
| seg_tumour_vol | 0.060 | 0.034 to 0.099 | 188 |

## healthy

| metric | median | IQR | n |
|---|---|---|---|
| det_drop_cam_10 | -0.000 | -0.000 to -0.000 | 86 |
| det_drop_cam_20 | -0.000 | -0.000 to -0.000 | 86 |
| det_drop_cam_30 | -0.015 | -0.453 to -0.000 | 86 |
| det_drop_random_10 | -0.000 | -0.000 to -0.000 | 86 |
| det_drop_random_20 | -0.000 | -0.000 to -0.000 | 86 |
| det_drop_random_30 | -0.000 | -0.000 to -0.000 | 86 |
| det_outside_mass | 0.583 | 0.574 to 0.597 | 86 |
| det_shell_mass | 0.078 | 0.076 to 0.079 | 86 |
| det_shell_ratio | 0.342 | 0.335 to 0.350 | 86 |
| det_shell_vol | 0.226 | 0.225 to 0.228 | 86 |
