# Grad-CAM++ checks (main, test, n=275)

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
| det_outside_mass | 0.702 | 0.687 to 0.722 | 188 |
| det_shell_mass | 0.058 | 0.056 to 0.060 | 188 |
| det_shell_ratio | 0.246 | 0.234 to 0.260 | 188 |
| det_shell_vol | 0.233 | 0.228 to 0.239 | 188 |
| det_tumour_mass | 0.022 | 0.013 to 0.032 | 188 |
| det_tumour_ratio | 0.342 | 0.276 to 0.407 | 188 |
| det_tumour_vol | 0.064 | 0.035 to 0.095 | 188 |
| seg_outside_mass | 0.219 | 0.145 to 0.319 | 188 |
| seg_shell_mass | 0.077 | 0.066 to 0.095 | 188 |
| seg_shell_ratio | 0.331 | 0.282 to 0.409 | 188 |
| seg_shell_vol | 0.233 | 0.228 to 0.239 | 188 |
| seg_tumour_mass | 0.546 | 0.409 to 0.651 | 188 |
| seg_tumour_ratio | 7.917 | 6.343 to 11.171 | 188 |
| seg_tumour_vol | 0.064 | 0.035 to 0.095 | 188 |

## healthy

| metric | median | IQR | n |
|---|---|---|---|
| det_drop_cam_10 | -0.000 | -0.000 to 0.000 | 87 |
| det_drop_cam_20 | -0.000 | -0.000 to -0.000 | 87 |
| det_drop_cam_30 | -0.006 | -0.271 to -0.000 | 87 |
| det_drop_random_10 | -0.000 | -0.000 to -0.000 | 87 |
| det_drop_random_20 | -0.000 | -0.000 to -0.000 | 87 |
| det_drop_random_30 | -0.000 | -0.000 to -0.000 | 87 |
| det_outside_mass | 0.583 | 0.571 to 0.593 | 87 |
| det_shell_mass | 0.078 | 0.077 to 0.080 | 87 |
| det_shell_ratio | 0.346 | 0.336 to 0.354 | 87 |
| det_shell_vol | 0.226 | 0.225 to 0.228 | 87 |
