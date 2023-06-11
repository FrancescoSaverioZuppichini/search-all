from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError
from PIL import Image
from tqdm import tqdm, trange

dataset = load_dataset("xfh/lexica_6k", split="train")

SAVE_DIR = "data/lexica"
BATCH_SIZE = 16
BUCKET_NAME = "activeloop-sandbox-fdd0"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def store_image(name: str, image: Image):
    image.save(f"{SAVE_DIR}/{name}.jpeg")


# append to bucket - this part is gpt + me generate because aws is the worst thing ever invented
def store_image_to_s3(file_path: Path, bucket: str):
    with file_path.open("rb") as data:
        try:
            s3.upload_fileobj(data, bucket, file_path.stem)
            print(f"Upload Successful for file: {file_path}")
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False


# first store them into a folder cuz I'll need them later and I don't want to change the pipeline
with ThreadPoolExecutor(max_workers=16) as executor:
    # for i in trange(0, len(dataset), BATCH_SIZE):
    #     rows = dataset[i: i + BATCH_SIZE]
    #     names = rows['text']
    #     images = rows['image']
    #     list(tqdm(executor.map(lambda x: store_image(*x), zip(names, images)), leave=False, total=BATCH_SIZE))

    print("[INFO] uploading to S3 ....")
    files_to_upload = list(Path(SAVE_DIR).glob("*.jpeg"))
    list(
        tqdm(
            executor.map(
                store_image_to_s3, files_to_upload, [BUCKET_NAME] * len(files_to_upload)
            )
        )
    )
