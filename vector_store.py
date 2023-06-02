import json
import os

import deeplake
import numpy as np
import torch
from deeplake.constants import MB
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.io.image import read_image
from multiprocessing import Pool
from functools import partial

class VectorStore:
    def __init__(self, dataset_path: str, token: str, org_id: str):
        self.dataset_path = dataset_path
        self._ds = deeplake.load(
            dataset_path, read_only=True, token=token
        )
        
    def search(self, embedding: np.ndarray):
        query = f"select * from (select cosine_similarity(embeddings - ARRAY{embedding[0].tolist()}) as score from \"{vs.dataset_path}\") order by score desc limit 5"
        query_res = self._ds.query(query)
        return query_res

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
            image = read_image(metadata["path"]).permute(1,2,0).numpy()
            metadata['path'] = Path(metadata['path']).name
            ds.append(
                {"embeddings": embedding, "metadata": metadata, "images": image}
            )

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
            org_id=org_id,
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

            vector_store = VectorStore(dataset_path, token, org_id)

            embeddings_data_paths = embeddings_root.glob("*.pth")
            list(tqdm(map(partial(vector_store.add_torch_embeddings, ds), embeddings_data_paths)))

            # with Pool(8) as p:
            #     list(
            #         tqdm(
            #             p.imap(partial(vector_store.add_torch_embeddings, ds), embeddings_data_paths)
            #         )
            #     )

        return vector_store


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

    # create_ds("embeddings/text3.pth")
