import argparse
import os
import random
import shutil
import sys


def clean_dir(path):
    removed = 0
    if os.path.isdir(path):
        for name in os.listdir(path):
            file_path = os.path.join(path, name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                removed += 1
    return removed


def resolve_src(src):
    candidates = [
        src,
        os.path.join('..', 'Kvasir-SEG'),
        os.path.join('..', 'U-Net-ASPP', 'kvasir-seg', 'Kvasir-SEG'),
        os.path.join('..', '..', 'U-Net-ASPP', 'kvasir-seg', 'Kvasir-SEG'),
    ]
    for candidate in candidates:
        if (os.path.isdir(os.path.join(candidate, 'images'))
                and os.path.isdir(os.path.join(candidate, 'masks'))):
            return candidate
    return src


def main():
    ap = argparse.ArgumentParser(description="Split Kvasir-SEG into data_root/train and data_root/test.")
    ap.add_argument('--src', default=os.path.join('..', 'Kvasir-SEG'),
                    help='Kvasir-SEG root containing images/ and masks/')
    ap.add_argument('--dst', default='data')
    ap.add_argument('--test-count', type=int, default=10,
                    help='number of test image/mask pairs')
    ap.add_argument('--val-ratio', type=float, default=None,
                    help='optional ratio override, used when --test-count is not given')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-clean', action='store_true',
                    help='keep existing files in target Img/GT directories')
    args = ap.parse_args()

    args.src = resolve_src(args.src)

    img_src = os.path.join(args.src, 'images')
    mask_src = os.path.join(args.src, 'masks')
    if not os.path.isdir(img_src) or not os.path.isdir(mask_src):
        sys.exit(f'images/ or masks/ not found under {os.path.abspath(args.src)}')

    mask_names = set(os.listdir(mask_src))
    names = sorted(n for n in os.listdir(img_src) if n in mask_names)
    if not names:
        sys.exit('no matched image/mask pairs found')

    random.Random(args.seed).shuffle(names)

    if args.val_ratio is not None:
        n_test = int(round(len(names) * args.val_ratio))
    else:
        n_test = args.test_count
    n_test = min(max(1, n_test), len(names) - 1)
    splits = {'test': names[:n_test], 'train': names[n_test:]}

    for split, files in splits.items():
        for sub in ('Img', 'GT'):
            target = os.path.join(args.dst, split, sub)
            os.makedirs(target, exist_ok=True)
            if not args.no_clean:
                removed = clean_dir(target)
                if removed:
                    print(f'cleaned {removed} old files in {target}')
        for name in files:
            shutil.copy2(os.path.join(img_src, name), os.path.join(args.dst, split, 'Img', name))
            shutil.copy2(os.path.join(mask_src, name), os.path.join(args.dst, split, 'GT', name))
        print(f'{split}: {len(files)} images/masks -> {os.path.join(args.dst, split)}')

    print(f'total: {len(names)} pairs (seed={args.seed}, test={len(splits["test"])}, train={len(splits["train"])})')


if __name__ == '__main__':
    main()
