from models.unet import UNet, DoubleLayerResBlock, MultiLayerConvDropoutNormNonlin


def resunet_lw_dec(
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = True,
    prediction: bool = False,
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        encoder_block=DoubleLayerResBlock,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=False,
    )

    return unet_model


def resunet(
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = False,
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        encoder_block=DoubleLayerResBlock,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        use_skip_connections=True,
    )

    return unet_model