import segmentation_models_pytorch as smp


def build_model(encoder_name='resnet50', encoder_weights='imagenet',
                in_channels=3, out_channels=1):
    """Build the segmentation-models-pytorch UNet.

    ``activation=None`` keeps the raw logits so the shared BCE-with-logits +
    Dice loss and the sigmoid thresholding in test/compare stay unchanged.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )
