import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset, encode_mask, get_preprocess_fn
from model.UNet import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean(dim=1).mean().item()


def iou_coefficient(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def load_model(args):
    model = build_model(args.encoder, encoder_weights=None,
                        in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    return model.to(device).eval()


def _prepare(image, gt, img_size):
    image = transforms.Resize((img_size, img_size))(image)
    gt = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)(gt)
    return image, gt


def test(args, preprocess_fn):
    img_dir = os.path.join(args.data_root, 'test', 'Img')
    gt_dir = os.path.join(args.data_root, 'test', 'GT')
    name = sorted(os.listdir(img_dir))[0]
    image_path = os.path.join(img_dir, name)
    gt_path = os.path.join(gt_dir, name)

    model = load_model(args)

    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        image, gt = _prepare(image, gt, args.img_size)

        x = preprocess_fn(np.array(image))
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        x = x.unsqueeze(0).to(device)
        y = torch.from_numpy(encode_mask(gt)).unsqueeze(0).to(device)

        prob = torch.sigmoid(model(x))[0, 0]
        pred = (prob > 0.5).cpu().numpy().astype(bool)

        rgb = np.array(image)
        overlay = rgb.copy()
        overlay[pred] = (0.45 * rgb[pred] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)

        Image.fromarray(overlay).save('result.jpg')
        Image.fromarray((pred * 255).astype(np.uint8)).save('result_mask.png')

        dice_score = dice_coefficient((prob > 0.5)[None, None].float(), y.float())
        print(f'Dice Coefficient {name}: {dice_score:.4f}')


def eval(args, preprocess_fn):
    model = load_model(args)
    test_dataset = MyDataset(os.path.join(args.data_root, 'test', 'Img'),
                             os.path.join(args.data_root, 'test', 'GT'),
                             img_size=args.img_size,
                             preprocessing_fn=preprocess_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for images, gts in test_loader:
            images = images.to(device)
            gts = gts.to(device)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            dice_scores.append(dice_coefficient(preds, gts))
            iou_scores.append(iou_coefficient(preds, gts))

    print(f'Average Dice: {np.mean(dice_scores):.4f}')
    print(f'Average IoU : {np.mean(iou_scores):.4f}')


def main():
    ap = argparse.ArgumentParser(description="Test a trained smp UNet checkpoint.")
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--weights', default='unet_smp-best.pth')
    ap.add_argument('--encoder', default='resnet50')
    ap.add_argument('--encoder-weights', default='imagenet',
                    help='weights tag used for preprocessing normalization')
    args = ap.parse_args()

    preprocess_fn = get_preprocess_fn(args.encoder, args.encoder_weights)
    test(args, preprocess_fn)
    eval(args, preprocess_fn)


if __name__ == '__main__':
    main()
