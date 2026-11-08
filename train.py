import os

import configs.config_loader as cfg_loader
from dataset import MyDataset, get_preprocess_fn
from model.UNet import build_model
from model.training import Trainer


def main():
    args = cfg_loader.get_config()
    model = build_model(args.encoder, args.encoder_weights)
    preprocess_fn = get_preprocess_fn(args.encoder, args.encoder_weights)

    train_dir = os.path.join(args.data_root, 'train')
    TrainDataset = MyDataset(os.path.join(train_dir, 'Img'),
                             os.path.join(train_dir, 'GT'),
                             img_size=args.img_size,
                             augment=True,
                             preprocessing_fn=preprocess_fn)

    trainer = Trainer(model, TrainDataset, args)
    trainer.train_model()


if __name__ == '__main__':
    main()
