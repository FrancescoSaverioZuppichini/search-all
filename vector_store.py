import json
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

import deeplake
import numpy as np
import torch
from deeplake.constants import MB
from torchvision.io.image import read_image
from tqdm import tqdm


class VectorStore:
    def __init__(self, dataset_path: str, token: str, org_id: str):
        self.dataset_path = dataset_path
        self._ds = deeplake.load(dataset_path, read_only=True, token=token)

    def retrieve(self, embedding: torch.Tensor, limit: int = 15) -> List[str]:
        query = f'select * from (select metadata, cosine_similarity(embeddings, ARRAY{embedding.tolist()}) as score from "{self.dataset_path}") order by score desc limit {limit}'
        query_res = self._ds.query(query, runtime={"tensor_db": True})
        images = [
            el["path"].split(".")[0]
            for el in query_res.metadata.data(aslist=True)["value"]
        ]
        return images, query_res

    @classmethod
    def from_env(cls):
        return cls(
            f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{os.environ['ACTIVELOOP_DATASET_ID']}",
            os.environ["ACTIVELOOP_TOKEN"],
            os.environ["ACTIVELOOP_ORG_ID"],
        )

    @staticmethod
    def add_torch_embeddings(ds: deeplake.Dataset, embeddings_data_path: Path):
        embeddings_data = torch.load(embeddings_data_path)
        for embedding_data in embeddings_data:
            metadata = embedding_data["metadata"]
            embedding = embedding_data["embedding"].cpu().float().numpy()
            image = read_image(metadata["path"]).permute(1, 2, 0).numpy()
            metadata["path"] = Path(metadata["path"]).name
            ds.append({"embeddings": embedding, "metadata": metadata, "images": image})

    @classmethod
    def from_torch_embeddings(
        cls,
        embeddings_root: Path,
        dataset_path: str,
        token: str,
        org_id: str,
        overwrite: bool = True,
    ):
        ds = deeplake.empty(
            path=dataset_path,
            runtime={"db_engine": True},
            token=token,
            overwrite=overwrite,
            # org_id=org_id,
        )

        with ds:
            ds.create_tensor(
                "metadata",
                htype="json",
                create_id_tensor=False,
                create_sample_info_tensor=False,
                create_shape_tensor=False,
                chunk_compression="lz4",
            )
            ds.create_tensor("images", htype="image", sample_compression="jpg")
            ds.create_tensor(
                "embeddings",
                htype="embedding",
                dtype=np.float32,
                create_id_tensor=False,
                create_sample_info_tensor=False,
                max_chunk_size=64 * MB,
                create_shape_tensor=True,
            )

            embeddings_data_paths = embeddings_root.glob("*.pth")
            list(
                tqdm(
                    map(
                        partial(VectorStore.add_torch_embeddings, ds),
                        embeddings_data_paths,
                    )
                )
            )

        return VectorStore(dataset_path, token, org_id)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = (
        f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{os.environ['ACTIVELOOP_DATASET_ID']}"
    )
    token = os.environ["ACTIVELOOP_TOKEN"]
    org_id = os.environ["ACTIVELOOP_ORG_ID"]
    store = VectorStore.from_torch_embeddings(
        Path("embeddings/"), dataset_path=dataset_path, token=token, org_id=org_id
    )
