from sys import path

path.append(".")

from dotenv import load_dotenv

load_dotenv()

from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from models.model_utils import get_images_embeddings, get_model


def chunks(iterator, batch_size):
    while chunk := list(islice(iterator, batch_size)):
        yield chunk


@torch.no_grad()
def encode_images(
    images_root: Path,
    model: torch.nn.Module,
    embeddings_out_dir: Path,
    batch_size: int = 64,
):
    # not the best way but the faster, best way would be to use a torch Dataset + Dataloader
    images = images_root.glob("*.jpeg")
    embeddings_out_dir.mkdir(exist_ok=True)
    for batch_idx, chunk in tqdm(enumerate(chunks(images, batch_size))):
        images_paths_str = [str(el) for el in chunk]
        images_embeddings = get_images_embeddings(model, images_paths_str)
        torch.save(
            [
                {"metadata": {"path": image_path}, "embedding": embedding}
                for image_path, embedding in zip(images_paths_str, images_embeddings)
            ],
            f"{str(embeddings_out_dir)}/{batch_idx}.pth",
        )


if __name__ == "__main__":
    from models.model_utils import device

    images_root = Path("data/lexica")
    model = get_model().half()
    encode_images(images_root, model, Path("embeddings/"), device)
