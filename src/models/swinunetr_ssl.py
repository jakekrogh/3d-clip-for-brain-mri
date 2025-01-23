# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import copy
import lightning as L

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep



class SwinUNETR_SSL(L.LightningModule):
    def freeze(self):

        for param in list(self.swinViT.layers1.parameters()):
            param.requires_grad = False
  
        for param in list(self.swinViT.layers2.parameters()):
            param.requires_grad = False          
    def __init__(
        self,
        contrastive: bool = False,
        rotation: bool = False,
        reconstruction: bool = False,
        input_channels: int = 1,
        num_classes: int = 1,
        enc_out_dim=768,
        upsample="vae",
        use_grad_checkpointing=False,
        dropout_path_rate=0.0,
        spatial_dims=3,
    ):
        super().__init__()

        if not (contrastive or rotation or reconstruction):
            print("Instantiating SwinUNETR in headless mode.")

        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        embed_dim = enc_out_dim // 16  # checked
        self.enc_out_dim = enc_out_dim
        self.swinViT = SwinViT(
            in_chans=input_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_grad_checkpointing,
            spatial_dims=spatial_dims,
        )  # .forward returns [x0_out, x1_out, x2_out, x3_out, x4_out] => x4_out is the last layer

        # self.contrastive = contrastive
        # self.rotation = rotation
        # self.reconstruction = reconstruction
        # if self.rotation:
        #     self.rotation_pre = nn.Identity()
        #     self.rotation_head = nn.Linear(enc_out_dim, 4)

        # if self.contrastive:
        #     self.contrastive_pre = nn.Identity()
        #     self.contrastive_head = nn.Linear(enc_out_dim, 512)

        # if self.reconstruction:
        #     if upsample == "large_kernel_deconv":
        #         self.rec_head = nn.ConvTranspose3d(enc_out_dim, num_classes, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        #     elif upsample == "deconv":
        #         self.rec_head = nn.Sequential(
        #             nn.ConvTranspose3d(enc_out_dim, enc_out_dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #             nn.ConvTranspose3d(enc_out_dim // 2, enc_out_dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #             nn.ConvTranspose3d(enc_out_dim // 4, enc_out_dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #             nn.ConvTranspose3d(enc_out_dim // 8, enc_out_dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #             nn.ConvTranspose3d(enc_out_dim // 16, num_classes, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #         )
        #     elif upsample == "vae":
        #         self.rec_head = nn.Sequential(
        #             nn.Conv3d(enc_out_dim, enc_out_dim // 2, kernel_size=3, stride=1, padding=1),
        #             nn.InstanceNorm3d(enc_out_dim // 2),
        #             nn.LeakyReLU(),
        #             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #             nn.Conv3d(enc_out_dim // 2, enc_out_dim // 4, kernel_size=3, stride=1, padding=1),
        #             nn.InstanceNorm3d(enc_out_dim // 4),
        #             nn.LeakyReLU(),
        #             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #             nn.Conv3d(enc_out_dim // 4, enc_out_dim // 8, kernel_size=3, stride=1, padding=1),
        #             nn.InstanceNorm3d(enc_out_dim // 8),
        #             nn.LeakyReLU(),
        #             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #             nn.Conv3d(enc_out_dim // 8, enc_out_dim // 16, kernel_size=3, stride=1, padding=1),
        #             nn.InstanceNorm3d(enc_out_dim // 16),
        #             nn.LeakyReLU(),
        #             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #             nn.Conv3d(enc_out_dim // 16, enc_out_dim // 16, kernel_size=3, stride=1, padding=1),
        #             nn.InstanceNorm3d(enc_out_dim // 16),
        #             nn.LeakyReLU(),
        #             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #             nn.Conv3d(enc_out_dim // 16, num_classes, kernel_size=1, stride=1),
        #         )
        #     else:
        #         raise ValueError(upsample)
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
    def forward(self, x, return_seq = False):
        x_out = self.swinViT(x.contiguous())[4]

        avg = self.adaptive_avg_pooling(x_out)
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        if return_seq:
            return avg, x4_reshape
        else:
            return avg
        # _, c, h, w, d = x_out.shape

        # x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        # x4_reshape = x4_reshape.transpose(1, 2)
        # print(x_out.size())
        # print(x_out_avg.size())
        # print(x4_reshape.size())
        # if self.rotation:
        #     x_rot = self.rotation_pre(x4_reshape[:, 0])
        #     x_rot = self.rotation_head(x_rot)
        # else:
        #     x_rot = None

        # if self.contrastive:
        #     x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        #     x_contrastive = self.contrastive_head(x_contrastive)
        # else:
        #     x_contrastive = None

        # x_rec = x_out.flatten(start_dim=2, end_dim=4)
        # x_rec = x_rec.view(-1, c, h, w, d)
        # x_rec = self.rec_head(x_rec)

        # return x_contrastive, x_rec, x_rot
        

    def load_state_dict(self, state_dict, *args, **kwargs):
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k[:6] == 'model.':
                name = k[6:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict
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


def swinunetr_ssl(
    input_channels: int = 1,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
):
    return SwinUNETR_SSL(
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        input_channels=input_channels,
        num_classes=num_classes,
    )
