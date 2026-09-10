"""
Helper script to generate starter Jupyter Notebooks.
"""

from pathlib import Path
import nbformat as nbf

def generate_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(exist_ok=True)

    # 1. 01_data_preprocessing.ipynb
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell("# 01. Data Ingestion & Preprocessing\nAutomated calculation of NDWI, MNDWI, cloud/shadow masking, and SAR Lee speckle filtering."),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src.preprocessing.compute_indices import compute_ndwi, compute_mndwi\n"
            "from src.dataset.build_input_stack import create_synthetic_glacial_scene\n\n"
            "# Generate sample multi-band scene\n"
            "stack, mask, prof = create_synthetic_glacial_scene(height=512, width=512, epoch=2016)\n"
            "print('Input Stack Shape:', stack.shape)\n"
            "print('Lake Mask Positive Pixels:', np.sum(mask > 0))"
        ),
        nbf.v4.new_code_cell(
            "fig, axs = plt.subplots(1, 3, figsize=(15, 5))\n"
            "axs[0].imshow(stack[2], cmap='gray'); axs[0].set_title('Red Band B04')\n"
            "axs[1].imshow(stack[5], cmap='coolwarm'); axs[1].set_title('Normalized Difference Water Index NDWI')\n"
            "axs[2].imshow(mask, cmap='Blues'); axs[2].set_title('Ground Truth Lake Mask')\n"
            "plt.tight_layout(); plt.show()"
        )
    ]
    nbf.write(nb1, nb_dir / "01_data_preprocessing.ipynb")

    # 2. 02_lake_segmentation.ipynb
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell("# 02. Lake Segmentation Model Training\nTraining Multi-band U-Net and RGB Fallback Models with Combined Dice + BCE Loss."),
        nbf.v4.new_code_cell(
            "import torch\n"
            "from src.segmentation.unet_model import LakeUNet\n"
            "from src.segmentation.losses import BCEDiceLoss\n\n"
            "model = LakeUNet.create_multiband_model(n_channels=7)\n"
            "print('Model Total Parameters:', sum(p.numel() for p in model.parameters()))\n"
            "x = torch.randn(2, 7, 256, 256)\n"
            "logits = model(x)\n"
            "print('Output Logits Shape:', logits.shape)"
        ),
        nbf.v4.new_code_cell(
            "loss_fn = BCEDiceLoss(dice_weight=0.5, bce_weight=0.5)\n"
            "targets = torch.zeros(2, 1, 256, 256)\n"
            "loss = loss_fn(logits, targets)\n"
            "print('Initial Loss:', loss.item())"
        )
    ]
    nbf.write(nb2, nb_dir / "02_lake_segmentation.ipynb")

    # 3. 03_change_detection.ipynb
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell("# 03. Multi-Temporal Matching & Change Classification\nSpatial polygon matching (IoU + Centroid proximity) and classifying Newly Formed, Survived, and Drained lakes."),
        nbf.v4.new_code_cell(
            "from src.dataset.build_input_stack import create_synthetic_glacial_scene\n"
            "from src.postprocessing.vectorize import mask_to_geodataframe\n"
            "from src.change_detection.classify_change import classify_temporal_change, generate_change_summary_report\n\n"
            "_, mask_16, prof_16 = create_synthetic_glacial_scene(512, 512, seed=42, epoch=2016)\n"
            "_, mask_22, prof_22 = create_synthetic_glacial_scene(512, 512, seed=100, epoch=2022)\n\n"
            "gdf_16 = mask_to_geodataframe(mask_16, prof_16['transform'])\n"
            "gdf_22 = mask_to_geodataframe(mask_22, prof_22['transform'])\n\n"
            "classified_gdf, summary_df = classify_temporal_change(gdf_16, gdf_22)\n"
            "print('Summary Report:', generate_change_summary_report(summary_df))\n"
            "summary_df.head()"
        )
    ]
    nbf.write(nb3, nb_dir / "03_change_detection.ipynb")

    # 4. 04_visualization.ipynb
    nb4 = nbf.v4.new_notebook()
    nb4.cells = [
        nbf.v4.new_markdown_cell("# 04. Cartographic Map Generation & 3-Panel Scientific Visuals\nReplicating scientific literature figures (Panels a, b, c)."),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src.dataset.build_input_stack import create_synthetic_glacial_scene\n"
            "from src.inference.overlay import create_dual_epoch_overlay, draw_classified_markers, create_publication_three_panel_figure\n"
            "from src.postprocessing.vectorize import mask_to_geodataframe\n"
            "from src.change_detection.classify_change import classify_temporal_change\n\n"
            "stack_16, mask_16, prof = create_synthetic_glacial_scene(600, 600, seed=42, epoch=2016)\n"
            "stack_22, mask_22, _ = create_synthetic_glacial_scene(600, 600, seed=100, epoch=2022)\n\n"
            "rgb_22 = np.stack([stack_22[2], stack_22[1], stack_22[0]], axis=-1)\n"
            "rgb_22 = np.clip(rgb_22 * 255.0, 0, 255).astype(np.uint8)\n\n"
            "gdf_16 = mask_to_geodataframe(mask_16, prof['transform'])\n"
            "gdf_22 = mask_to_geodataframe(mask_22, prof['transform'])\n"
            "_, summary_df = classify_temporal_change(gdf_16, gdf_22)\n\n"
            "panel_b = create_dual_epoch_overlay(rgb_22, mask_16, mask_22)\n"
            "panel_c = draw_classified_markers(rgb_22, summary_df.to_dict(orient='records'))\n"
            "fig = create_publication_three_panel_figure(rgb_22, panel_b, panel_c, output_path='outputs/maps/three_panel_visual.png')\n"
            "plt.show()"
        )
    ]
    nbf.write(nb4, nb_dir / "04_visualization.ipynb")

    print("All 4 Jupyter Notebooks generated successfully.")

if __name__ == "__main__":
    generate_notebooks()
