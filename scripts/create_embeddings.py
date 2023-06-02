from sys import path

path.append(".")

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from itertools import islice

import torch

import models.data as data
from models import imagebind_model
from models.imagebind_model import ModalityType
from models.model_utils import get_model, get_images_embeddings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def chunks(iterator, batch_size):
    while chunk := list(islice(iterator, batch_size)):
        yield chunk


@torch.no_grad()
def encode_images(
    images_root: Path,
    model: torch.nn.Module,
    embeddings_out_dir: Path,
    device: torch.device,
    batch_size: int = 64,
):
    # not the best way but the faster, best way would be to use a torch Dataset + Dataloader
    images = images_root.glob("*.jpg")
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

    images_root = Path(
        "/home/zuppif/Documents/Work/ActiveLoop/search-all/data/coco_minitrain_25k/images/val2017"
    )
    model = get_model().half()
    encode_images(images_root, model, Path("embeddings/"), device)
