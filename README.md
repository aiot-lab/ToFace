# ToFace: Training Scripts

## Step 1: Train the Face Localization Model
This step trains the face localization model to generate masks for the input data.

```bash
python To_Face_maskGenerate_Train.py \
    --data_folder inves_dataset/dataset_FER/ \
    --model New_ssd \
    --data_type rd
```







**Arguments:**
- `--data_folder` : Path to the dataset folder (e.g., `inves_dataset/dataset_FER/`).
- `--model`       : Model type to use (e.g., `New_ssd`).
- `--data_type`   : Type of data input.

---

## Step 2: Train the Structured Reconstruction & Expression Classification Model
This step trains the model for structured reconstruction and expression classification.

```bash
python To_Face_run_FER.py \
    --data_folder inves_dataset/dataset_FER/ \
    --model ToFace_Unet_cnn \
    --out_size 32
```

**Arguments:**
- `--data_folder` : Path to the dataset folder (e.g., `inves_dataset/dataset_FER/`).
- `--model`       : Model type to use (e.g., `ToFace_Unet_cnn`).
- `--out_size`    : Output size for structured reconstruction (e.g., `32` for 32x32 resolution).
```