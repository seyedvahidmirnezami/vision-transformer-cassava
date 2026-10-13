# Fine-Tune a Vision Transformer on Cassava Leaves

[![Course on Udemy](https://img.shields.io/badge/Full%20course-Udemy-A435F0?logo=udemy&logoColor=white)](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)
[![Kaggle competition](https://img.shields.io/badge/Kaggle-Cassava%20Leaf%20Disease-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/cassava-leaf-disease-classification)

Fine-tune a pretrained Vision Transformer (`google/vit-base-patch16-224`) to classify cassava leaf photos into 5 disease classes,
then write a Kaggle submission, all in one notebook that runs on a free Kaggle T4 GPU.

## How a Vision Transformer sees an image

A ViT reuses the transformer encoder from NLP almost unchanged. The only new idea is how an image becomes a sequence:

1. Cut the 224×224 image into 16×16 **patches** (196 of them).
2. Flatten each patch and project it to a vector: these are the "tokens".
3. Add a learnable **[CLS]** token and **position embeddings**.
4. Run the standard transformer encoder: self-attention between patches.
5. Classify from the final [CLS] vector.

Fine-tuning swaps the original 1,000-class ImageNet head for a new 5-class cassava head and trains on leaf photos.

## What the notebook does

| Step | What happens |
|---|---|
| 1 | Setup and imports |
| 2 | Find the competition folder and load labels |
| 3 | Explore class balance and view a leaf image |
| 4 | Stratified train/validation split, with an optional smaller classroom sample |
| 5 | Load the ViT image processor |
| 6 | Build Hugging Face image datasets |
| 7 | Load pretrained ViT with a new 5-class head |
| 8 | Metrics, `TrainingArguments` and `Trainer` |
| 9 | Fine-tune and evaluate |
| 10 | Predict on test images and write `submission.csv` |

## Run it on Kaggle

1. Open the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) competition and accept the rules.
2. From the competition's **Code** tab, create a new notebook and upload [`cassava_vit.ipynb`](cassava_vit.ipynb) (**File → Import Notebook**).
3. **Session options → Accelerator → GPU T4**, and **Internet → On** so the ViT weights can download.
4. **Run All**.

By default the notebook trains on a stratified sample of 4,000 images for 2 epochs so it finishes quickly.
Set `MAX_TRAIN = None` to train on all ~21k labeled images.

## Ideas to go further

- Train on the full dataset with more epochs
- Add stronger augmentation (for example Albumentations)
- Ensemble ViT with a CNN, as many top Cassava solutions did

## Learn the full transformer

This is Session 15 of **[Master Transformers from Scratch: BERT, ViT & AI Search](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)**.
The course builds up to this from the math of attention, through the encoder block and BERT, so you understand every layer you fine-tune here.

More from the course:
- [transformer-encoder-pytorch](https://github.com/seyedvahidmirnezami/transformer-encoder-pytorch): the full encoder block, and why positional encoding matters
- [self-attention-from-scratch](https://github.com/seyedvahidmirnezami/self-attention-from-scratch): attention with real numbers, in NumPy and PyTorch
- [semantic-search-two-tower](https://github.com/seyedvahidmirnezami/semantic-search-two-tower): build two-tower retrieval from scratch

If this helped you, a ⭐ helps other learners find it.

## License

[MIT](LICENSE). The cassava dataset is subject to the Kaggle competition rules.
