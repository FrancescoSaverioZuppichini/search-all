from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from models.model_utils import (
    ModalityType,
    get_images_embeddings,
    get_model,
    get_texts_embeddings,
)


def search(texts):
    text_embedding = get_texts_embeddings(model, texts)
    scores = torch.softmax(text_embedding @ embeddings.T, dim=1)
    return scores


def offset_image(coord, path, ax):
    img = Image.open(path)
    img.thumbnail((80, 80))
    im = OffsetImage(np.array(img))
    im.image.axes = ax

    ab = AnnotationBbox(
        im,
        (coord, 0),
        xybox=(0.0, -16.0),
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0,
    )

    ax.add_artist(ab)


def make_heatmap(texts, images_paths, scores):
    fig, ax = plt.subplots(figsize=(11, 8), dpi=400)
    fig.subplots_adjust(left=0.15)
    sns.heatmap(
        scores,
        annot=True,
        ax=ax,
        yticklabels=texts,
        xticklabels=False,
        cbar_kws={"label": "Similarity Score"},
    )
    plt.title("Imagebind embeddings vs text queries", fontsize=16)

    for item in ax.get_yticklabels():
        item.set_rotation(0)
        item.set_size(14)

    for i, file in enumerate(images_paths):
        img = Image.open(file)
        img.thumbnail((160, 160))  # Set thumbnail size
        ax_image = fig.add_axes(
            [i * 0.165 + 0.12, -0.05, 0.16, 0.2]
        )  # adjust these values to position your images
        ax_image.axis("off")  # hide the axes
        ax_image.imshow(np.array(img))  # show the image

    return fig


images_paths = list(
    Path("/home/zuppif/Documents/Work/ActiveLoop/search-all/data/text").glob("*")
)
images_paths = [
    Path("data/text/1.png"),
    Path("data/text/2.png"),
    Path("data/text/3.png"),
    Path("data/text/4.png"),
]
model = get_model()
embeddings = get_images_embeddings(model, [str(el) for el in images_paths])
texts = ["It's over", "Anakin", "I have", "the high ground"]
scores = search(texts)
make_heatmap(texts, [str(el) for el in images_paths], scores.cpu().numpy())

plt.savefig("my_figure.png")
