# Models Directory

This directory contains the trained models.

## Training Your Own Model

To train a model:

```bash
python train.py --epochs 20 --batch_size 32
```

After training, you will get:
- `best_model.pth` - The trained model (129MB)
- `metrics.json` - Performance metrics

## Note

Trained models are not included in the repository due to their large size (100+ MB).
You need to train your own model following the instructions in the main README.

## Pre-trained Models

For quick testing, you can download a pre-trained model from:
- [Releases page](../../releases) (if available)
- Or train your own following the README instructions
