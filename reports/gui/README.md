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
