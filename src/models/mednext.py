import copy
from re import I
from typing import Optional, Union
import torch.nn as nn
from yucca.networks.blocks_and_layers.conv_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock,
)
from yucca.networks.networks.YuccaNet import YuccaNet


class MedNeXt(YuccaNet):
    """
    From the paper: https://arxiv.org/pdf/2303.09975.pdf
    code source: https://github.com/MIC-DKFZ/MedNeXt/tree/main
    """
    def freeze(self):

        for param in list(self.encoder.enc_block_0.parameters()):
            param.requires_grad = False
  
        for param in list(self.encoder.enc_block_1.parameters()):
            param.requires_grad = False          
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        contrastive: bool = False,
        rotation: bool = False,
        reconstruction: bool = False,
        prediction: bool = False,
        conv_op=nn.Conv3d,
        starting_filters: int = 32,
        enc_exp_r: Union[int, list] = 2,
        dec_exp_r: Union[int, list] = 2,
        kernel_size: int = 5,
        dec_kernel_size: Optional[int] = None,
        deep_supervision: bool = False,
        do_res: bool = True,
        do_res_up_down: bool = True,
        enc_block_counts: list = [2, 2, 2, 2, 2],
        dec_block_counts: list = [2, 2, 2, 2],
        norm_type="group",
        grn=False,
    ):
        super().__init__()
        self.contrastive = contrastive
        self.rotation = rotation
        self.reconstruction = reconstruction
        self.prediction = prediction
        print(
            f"Loaded MedNeXt with Contrastive: {self.contrastive}, Rotation: {self.rotation}, Reconstruction: {self.reconstruction}, Prediction: {self.prediction}"
        )

        dim = starting_filters * 16
        self.dim = 512
        self.num_classes = output_channels

        self.encoder = MedNeXtEncoder(
            input_channels=input_channels,
            conv_op=conv_op,
            starting_filters=starting_filters,
            kernel_size=kernel_size,
            exp_r=enc_exp_r,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=enc_block_counts,
            norm_type=norm_type,
            grn=grn,
        )
        if self.contrastive:
            self.con_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(dim, 512))

        if self.rotation:
            self.rot_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(dim, 4), nn.Softmax(dim=1))

        if self.reconstruction:
            self.rec_head = MedNeXtDecoder(
                output_channels=output_channels,
                starting_filters=starting_filters,
                kernel_size=kernel_size,
                exp_r=dec_exp_r,
                dec_kernel_size=dec_kernel_size,
                deep_supervision=deep_supervision,
                do_res=do_res,
                do_res_up_down=do_res_up_down,
                block_counts=dec_block_counts,
                norm_type=norm_type,
                grn=grn,
            )

        if self.prediction:
            self.pred_head = MedNeXtDecoder(
                output_channels=output_channels,
                starting_filters=starting_filters,
                kernel_size=kernel_size,
                exp_r=dec_exp_r,
                dec_kernel_size=dec_kernel_size,
                deep_supervision=deep_supervision,
                do_res=do_res,
                do_res_up_down=do_res_up_down,
                block_counts=dec_block_counts,
                norm_type=norm_type,
                grn=grn,
            )
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool3d(output_size=(1,1,1))

    def forward(self, x):
        enc = self.encoder(x)
        # if self.prediction:
        #     return self.pred_head(enc)

        # y_hat_rot = self.rot_head(enc[4]) if self.rotation else None
        # y_hat_con = self.con_head(enc[4]) if self.contrastive else None
        # y_hat_rec = self.rec_head(enc) if self.reconstruction else None

        return self.adaptive_avg_pooling(enc[-1])

    def parameter_index_map(self):
        assert self.prediction, "Layer wise lr is only implemented for prediction"
        # return the index of each parameter from k (encoder start) to 0 (decoder output)
        return {
            "encoder.stem": 10,
            "encoder.enc_block_0": 9,
            "encoder.down_0": 8,
            "encoder.enc_block_1": 8,
            "encoder.down_1": 7,
            "encoder.enc_block_2": 7,
            "encoder.down_2": 6,
            "encoder.enc_block_3": 6,
            "encoder.down_3": 5,
            "encoder.bottleneck": 5,
            "ds_out_conv0": 4,
            "pred_head.upsample1": 4,
            "pred_head.decoder_conv1": 4,
            "ds_out_conv1": 3,
            "pred_head.upsample2": 3,
            "pred_head.decoder_conv2": 3,
            "ds_out_conv2": 2,
            "pred_head.upsample3": 2,
            "pred_head.decoder_conv3": 2,
            "ds_out_conv3": 1,
            "pred_head.upsample4": 1,
            "pred_head.decoder_conv4": 1,
            "ds_out_conv4": 0,
            "pred_head.out_conv": 0,
        }



class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_op=nn.Conv3d,
        starting_filters: int = 32,
        exp_r: Union[int, list] = [3, 4, 8, 8, 8],
        kernel_size: int = 5,
        do_res: bool = True,
        do_res_up_down: bool = True,
        block_counts: list = [3, 4, 8, 8, 8],
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        dim = "3d"

        self.stem = conv_op(input_channels, starting_filters, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[0],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[1],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[2],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[3],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * starting_filters,
            out_channels=16 * starting_filters,
            exp_r=exp_r[4],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 16,
                    out_channels=starting_filters * 16,
                    exp_r=exp_r[4],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        x_res_0 = self.enc_block_0(x)

        x = self.down_0(x_res_0)

        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)

        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)

        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        return [x_res_0, x_res_1, x_res_2, x_res_3, x]


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 32,
        exp_r: Union[int, list] = [3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size: int = 5,
        dec_kernel_size: Optional[int] = None,
        deep_supervision: bool = False,
        do_res: bool = True,
        do_res_up_down: bool = True,
        block_counts: list = [8, 8, 4, 3],
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.output_channels = output_channels
        if kernel_size is not None:
            dec_kernel_size = kernel_size

        dim = "3d"

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[0],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[0],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[1],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[2],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[3],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[3],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.output_channels, dim=dim)

        if self.deep_supervision:
            raise NotImplementedError

        self.block_counts = block_counts

    def forward(self, xs: list):
        # unpack the output of the encoder
        x_res_0, x_res_1, x_res_2, x_res_3, x = xs

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)

        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)

        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        return x


class MedNeXtDecoderSSL(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        dec_kernel_size: Optional[int] = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        block_counts: list = [3, 4, 8, 8, 8, 8, 8, 4, 3],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.output_channels = output_channels

        if kernel_size is not None:
            dec_kernel_size = kernel_size

        dim = "3d"

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.output_channels, dim=dim)

        if self.deep_supervision:
            raise NotImplementedError

        self.block_counts = block_counts

    def forward(self, x: list):
        x = self.up_3(x)
        x = self.dec_block_3(x)

        x = self.up_2(x)
        x = self.dec_block_2(x)

        x = self.up_1(x)
        x = self.dec_block_1(x)

        x = self.up_0(x)
        x = self.dec_block_0(x)

        x = self.out_0(x)

        return x
    def load_state_dict(self, state_dict, *args, **kwargs):
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v for k, v in state_dict.items() if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape]
        rejected_keys_data = []

        # Here there's also potential to implement custom loading functions.
        # E.g. to load 2D pretrained models into 3D by repeating or something like that.

        # Now keep track of the # of layers with succesful weight transfers
        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(old_params.keys(), old_params.values(), new_params.values()):
            # If more than one param in layer is NE (not equal) to the original weights we've successfully loaded new weights.
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if param_name not in rejected_keys_new and param_name not in rejected_keys_shape:
                    rejected_keys_data.append(param_name)

        print(f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers")
        print(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

def mednext_s3(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = False,
):
    return MedNeXt(
        input_channels=input_channels,
        output_channels=num_classes,
        conv_op=conv_op,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        kernel_size=3,
        enc_exp_r=2,
        dec_exp_r=2,
        enc_block_counts=[2, 2, 2, 2, 2],
        dec_block_counts=[2, 2, 2, 2],
    )


# def mednext_s3_lw_dec(
#     input_channels: int,
#     num_classes: int = 1,
#     contrastive: bool = False,
#     rotation: bool = False,
#     reconstruction: bool = False,
# ):
#     net = mednext_s3(
#         input_channels=input_channels,
#         num_classes=num_classes,
#         contrastive=contrastive,
#         rotation=rotation,
#         reconstruction=reconstruction,
#         prediction=False,
#     )

#     net.rec_head = light_weight_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)

#     return net


# def mednext_s3_std_dec(
#     input_channels: int,
#     num_classes: int = 1,
#     contrastive: bool = False,
#     rotation: bool = False,
#     reconstruction: bool = False,
#     prediction: bool = True,
# ):
#     net = mednext_s3(
#         input_channels=input_channels,
#         num_classes=num_classes,
#         contrastive=contrastive,
#         rotation=rotation,
#         reconstruction=reconstruction,
#         prediction=prediction,
#     )

#     # sanity check
#     assert (prediction and not reconstruction) or (reconstruction and not prediction)

#     if reconstruction:
#         print("Using a standard unet decoder as reconstruction head")
#         net.rec_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)

#     if prediction:
#         print("Using a standard unet decoder as prediction head")
#         net.pred_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=True)

#     return net
