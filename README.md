# UNet (segmentation-models-pytorch) for Kvasir-SEG

segmentation-models-pytorch (**smp**) implementation of the UNet model from
[Gastrointestinal-Polyp-Segmentation](../Gastrointestinal-Polyp-Segmentation),
using pretrained ImageNet encoders (ResNet50 by default, MobileNetV2 optional).

This is the **smp counterpart** of the native
[`../UNet-Polyp-Segmentation`](../UNet-Polyp-Segmentation) repo. The two repos
share the same data pipeline, loss, optimizer and scripts, so only the model /
encoder differs and results are directly comparable.

## Model

- `smp.Unet` with `activation=None` (raw logits, so the BCE-with-logits + Dice
  loss is unchanged).
- Configurable encoder via `--encoder` (`resnet50`, `mobilenet_v2`, ...) and
  `--encoder_weights` (`imagenet` by default).
- Input normalization uses the encoder's own preprocessing parameters from
  `smp.encoders.get_preprocessing_fn`.

## Project Structure

```
UNet-Polyp-Segmentation-SMP/
├── configs/config_loader.py # CLI args incl. --encoder / --encoder_weights
├── model/
│   ├── UNet.py              # smp.Unet builder
│   ├── myloss.py            # BCE + Dice loss
│   └── training.py          # training loop / checkpoint saving
├── dataset.py               # Dataset + smp preprocessing helper
├── prepare_kvasir.py        # split raw Kvasir-SEG into data/train + data/test
├── train.py
├── test.py
├── compare.py
└── requirements.txt
```

## Dataset

```
data/
├── train/{Img,GT}
└── test/{Img,GT}
```

Build the split (defaults match the other repos: `seed=42`, `test-count=10`):

```bash
python prepare_kvasir.py --src ../Kvasir-SEG
```

## Train

```bash
# ResNet50 encoder (default)
python train.py --encoder resnet50 --epochs 30 --batch_size 4

# MobileNetV2 encoder
python train.py --encoder mobilenet_v2 --epochs 30 --batch_size 8
```

Checkpoints: `unet_smp-best.pth` and `unet_smp-epoch-{N}.pth`.

## Test

```bash
python test.py --encoder resnet50 --weights unet_smp-best.pth
```

Prints average Dice / IoU on the test split and writes `result.jpg` /
`result_mask.png`.

## Compare checkpoints

```bash
python compare.py --encoder resnet50 --model-step 5
```

## Notes

- The first run with `--encoder_weights imagenet` downloads pretrained weights
  into the Hugging Face cache.
- Pretrained encoders are heavily normalized; `dataset.py` applies the smp
  preprocessing function before converting to a tensor.
- Compared with the native repo, this version trains a pretrained encoder and
  should converge faster / score higher, but is not a pure architecture
  comparison unless the native baseline is trained for the same schedule.
