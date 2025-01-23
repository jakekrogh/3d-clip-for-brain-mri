from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Optional
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


FPS = 4
DPI = 100


def finish():
    wandb.finish()


def update_config(dct):
    wandb.config.update(dct)


def save_tensor(x, name):
    torch.save(x, name)
    wandb.save(name)
    print(f"Saved {name} to wandb")


def get_gif(y, slice_dim, version_dir, epoch, desc="", label=None, text=""):
    assert version_dir is not None
    assert epoch is not None
    path = f"{version_dir}/gifs"
    Path(path).mkdir(parents=False, exist_ok=True)

    figs = get_figs(y, slice_dim=slice_dim, n=None, desc=desc, label=label, text=text)
    frames = []
    arrays = []
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        buf.seek(0)
        frame_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        frame = cv2.imdecode(frame_arr, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.close(fig)
        arrays.append(frame)
        frames.append(Image.fromarray(frame))

    file_name = f"{path}/epoch_{epoch}.gif"
    # Create gif using some PIL magic
    np_arr = np.asarray(arrays)
    np.save(file_name+"arr", np_arr)
    frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    gif = wandb.Video(file_name, format="gif", fps=FPS)

    plt.close("all")

    return [gif]


def get_gif_local(y, slice_dim, save_dir, file_name, desc="", label=None, text=""):
    figs = get_figs(y, slice_dim=slice_dim, n=None, label=label, text=text, desc=desc)
    frames = []
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        buf.seek(0)
        frame_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        frame = cv2.imdecode(frame_arr, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.close(fig)
        frames.append(Image.fromarray(frame))

    file_name = f"{save_dir}/{file_name}.gif"
    # Create gif using some PIL magic
    frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    return

def get_imgs(y, slice_dim, n, desc=""):
    figs = get_figs(y, slice_dim=slice_dim, n=n, desc=desc)

    def wandb_img(fig):
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    plt.close("all")

    return [wandb_img(fig) for fig in figs]


def get_figs(y: torch.Tensor, slice_dim: int, n: Optional[int] = None, desc="", text="", label = None):
    assert len(y.shape) == 5, y.shape  # (B, 1, X, Y, Z)

    batch_idx = 0  # we always just plot the first batch element

    y = y[batch_idx].squeeze().detach().cpu()
    if label is not None:
        label = label[batch_idx].squeeze().detach().cpu()
        label = label.to(torch.float32)
    # we might use mixed precision, so we cast to ensure tensor is compatiable with numpy
    y = y.to(torch.float32)
    figs = []

    if n is None:
        index_range = torch.arange(y.shape[slice_dim])
    else:
        # we take the middle num_png_images images
        middle = y.shape[slice_dim] // 2
        index_range = torch.arange(middle - floor(n / 2), middle + ceil(n / 2))
        assert len(index_range) == n

    for i in index_range:
        yi = torch.index_select(y, slice_dim, i).squeeze()
        label_i = None
        if label is not None:
            label_i = torch.index_select(label, slice_dim, i).squeeze()


        fig = create_image(yi, rec_text=f"i: {i}", desc=desc, label=label_i, text=text)
        figs.append(fig)

    return figs


def create_image(yi, rec_text="", desc="", label = None, text=""):
    assert len(yi.shape) == 2, yi.shape

    if label is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), dpi=DPI)
        im = axes[0].imshow(yi, cmap="gray")
        label = axes[1].imshow(label, cmap="gray")
        fig.colorbar(im, ax=axes[0])
        fig.colorbar(label, ax=axes[1])
    else: 
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), dpi=DPI)

        im = axes.imshow(yi, cmap="gray")



    text = text.replace("\n", ' ')
    caption = desc + "\n" + text
    fig.text(0.5, 0.01, caption, ha="center")

    return fig