# GUI (Phase 8)

`streamlit run app/streamlit_app.py` with `data_local/main/best.pt` (`main`, epoch 32), the frozen
validation threshold in `data_local/main/threshold.json` (0.0168), and the 10 test cases exported by
`eval-main-r5-test` in `data_local/cases/` (already through the shared SynthStrip boundary).
Run locally on CPU on 1 Nov; all three views ran end to end without errors.

| file | what it shows |
|---|---|
| `1_tumour_segmentation.jpg` | BraTS2021_00809 (test): tumour probability 1.000, decision Tumour, volumes WT 188.3 / TC 60.7 / ET 53.7 mL, segmentation overlay |
| `2_tumour_segmentation_gradcam.jpg` | same case, segmentation Grad-CAM++ concentrated on the tumour |
| `3_healthy_detection_gradcam.jpg` | IXI071 (test): probability 0.000, No tumour; detection only (T1ce/FLAIR absent); the detection Grad-CAM++ peaks at the top of the head, not in the brain (see reports/xai_notes.md) |

The inference time shown in the app is a single wall-clock pass (the first one includes warm-up and
Grad-CAM++); the reported CPU time is measured separately with `src/cpu_timing.py`
(`reports/cpu_inference.json`, in `reports/tables.md`): 2.23 s +/- 0.03 s per case (detection +
segmentation, batch 1, after warm-up, 12-core AMD laptop CPU), against 0.107 s on a Tesla T4.

## PLAN2 view (4 Nov)

Model choice "PLAN2: per hemisphere" (default) with `data_local/p2_main/best.pt` (`p2-main`, epoch 35) and the frozen
val thresholds from `data_local/p2_main/results_val.json` (head 0.5, v* 1.544 mL). Taken after the PLAN2 test read
(`test_lock.json` 3 Nov 21:47), so showing bundled test cases reads nothing new.

| file | what it shows |
|---|---|
| `4_p2_tumour_hemispheres.jpg` | BraTS2021_00809: left hemisphere probability 0.015, no tumour (0 mL); right hemisphere 1.000, tumour, pass-B WT 186.5 mL; pass-A WT/TC/ET 187.8 / 61.5 / 53.4 mL; dashed line = midline cut |
| `5_p2_tumour_occlusion.jpg` | same case, right hemisphere, occlusion preview (16 mm stride, ~1-2 min on CPU) over the T1 slice |
