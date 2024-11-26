# ADET80k: Subtitle Dataset Generation and High-Quality Segmentation

This repository provides tools for generating a large-scale subtitle segmentation dataset (**ADET80k**) and training models for subtitle segmentation using DeepLabv3+ for coarse masks and Continuous Refinement Model (CRM) for high-quality segmentation.

---

## Repository Structure

```
.
├── examples                       # Example results for quick reference
├── GeneratingSubtitle             # Tools for generating the ADET80k dataset
│   ├── font_cn.ttf                # Font files for subtitles in different languages
│   ├── font_en.ttf
│   ├── font_ja.ttf
│   ├── font_ko.ttf
│   ├── images                     # Input images for dataset generation
│   ├── main.py                    # Main script for dataset generation
│   ├── masks                      # Generated masks (ground truth)
│   ├── my_dataset                 # Final compiled dataset for training
│   ├── text_ch.txt                # Subtitle corpus for Chinese
│   ├── text_en.txt                # Subtitle corpus for English
│   ├── text_ja.txt                # Subtitle corpus for Japanese
│   ├── text_ko.txt                # Subtitle corpus for Korean
│   └── usage.md                   # Detailed usage instructions for this folder
├── High-Quality-Segmentation      # High-quality segmentation pipeline
└── README.md                      # This file
```

---

## Workflow

### 1. Generate the Subtitle Dataset

#### Prepare the Subtitle Corpus
Modify the corpus for each language in:
- `text_en.txt` (English)
- `text_ch.txt` (Chinese)
- `text_ja.txt` (Japanese)
- `text_ko.txt` (Korean)

You can substitute the fonts for subtitles by replacing the font files:
- `font_en.ttf`, `font_cn.ttf`, `font_ja.ttf`, `font_ko.ttf`

#### Set Input Images and Run the Script
Specify the input image folder by modifying the `image_folder` variable in `main.py`.

Run the script to generate the dataset:
```bash
python3 main.py
```

This will generate the dataset into the `generated_dataset` folder:
- `1.jpg`: Image with subtitles
- `1.png`: Ground truth mask for segmentation

---

### 2. Compile the Dataset
Once the dataset is generated, organize it for training and validation:
- Use `./train` for training data and `./validate` for validation data.
- Rename the dataset folder as `my_dataset` and move it to the `High-Quality-Segmentation` folder:
  ```bash
  mv my_dataset ../High-Quality-Segmentation
  ```

---

### 3. Train DeepLabv3+ for Coarse Masks

Navigate to the segmentation folder:
```bash
cd High-Quality-Segmentation/Segmentation
```

Train the DeepLabv3+ model to produce coarse masks:
```bash
python main.py
```

Model checkpoints will be saved in:
```bash
High-Quality-Segmentation/Segmentation/checkpoints
```

---

### 4. Train the Refinement Model

Train the refinement model using the Continuous Refinement Model (CRM):
```bash
python train.py Exp_ID -i 230000 -b 12 --steps 22500 37500 --lr 1.15e-4 --ce_weight 1.0 --l1_weight 0.5 --l2_weight 0.5 --grad_weight 2.0
```

This step refines the coarse masks to achieve high-resolution and precise segmentation.

---

### 5. Perform Inference with Combined Models

Run inference using both models to generate segmentation results:
```bash
python combined_inference.py --dir 3b1b_all_language --model weights/Exp_ID_2024-10-30_03:29:48/model_150000 --output all_language_demo --clear
```

**Options**:
- `--dir`: Path to the input image folder.
- `--model`: Path to the trained model weights.
- `--output`: Folder for storing output results.
- `--clear`: Clears intermediate files after inference.

---

## Example Results
Below are examples of generated subtitles and their corresponding refined masks:

1. **English Subtitle**:
   - Original Image: `examples/english_original.jpg`
   - Refined Mask: `examples/english_refined.jpg`

2. **Japanese Subtitle**:
   - Original Image: `examples/japanese_original.jpg`
   - Refined Mask: `examples/japanese_refined.jpg`

---

## Requirements
- Python 3.8+
- Dependencies:
  - `Pillow`
  - `torch`
  - `torchvision`
  - Any additional dependencies (install via `requirements.txt` if provided).

---

Feel free to reach out if you encounter any issues or have questions!
