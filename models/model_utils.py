from typing import List, Union, Optional, Dict

import torch
import numpy as np
from PIL import Image
from .data import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data
from .imagebind_model import ModalityType, imagebind_huge

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device= "cpu"
ImageLike = Union["str", np.ndarray, Image.Image]


def get_model(dtype: torch.dtype = torch.float16) -> torch.nn.Module:
    model = imagebind_huge(pretrained=True)
    model = model.eval().to(device, dtype=dtype)
    return model


@torch.no_grad()
def get_texts_embeddings(model: torch.nn.Module, texts: List[str]) -> torch.Tensor:
    inputs = {ModalityType.TEXT: load_and_transform_text(texts, device)}
    texts_embeddings = model(inputs)[ModalityType.TEXT]
    return texts_embeddings


@torch.no_grad()
def get_images_embeddings(
    model: torch.nn.Module, images: List[ImageLike],dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    inputs = {ModalityType.VISION: load_and_transform_vision_data(images, device).to(dtype)}
    images_embeddings = model(inputs)[ModalityType.VISION]
    return images_embeddings


@torch.no_grad()
def get_embeddings(
    model: torch.nn.Module,
    texts: Optional[List[str]],
    images: Optional[List[ImageLike]],
    audio: Optional[List[str]],
    dtype: torch.dtype = torch.float16
) -> Dict[str, torch.Tensor]:  
    inputs = {}
    if texts is not None:
        # they need to be ints
        inputs[ModalityType.TEXT] = load_and_transform_text(texts, device)
    if images is not None:
        inputs[ModalityType.VISION] = load_and_transform_vision_data(images, device, dtype)
    if audio is not None:
        inputs[ModalityType.AUDIO] = load_and_transform_audio_data(audio, device, dtype)
    embeddings = model(inputs)
    return embeddings
