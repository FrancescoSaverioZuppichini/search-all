# Search All: Cross Modal Retrieval 📜🎵📷
You can search a collection of images using `text`, `images` or `audio`.

For this app, we have embedded stable diffusion generated images from [`lexica`](ttps://lexica.art/n) eval set, you can now search them using **text, images or audio**. The embeddings are generated using Meta [ImageBind](https://imagebind.metademolab.com/) a new powerful model capable of handling a lots of modalities.

## Installation

You have the same installation requirements as the original [imagebind](https://github.com/facebookresearch/ImageBind) plus a couple of more packages. You can install them all by

```python
conda create --name imagebind python=3.8 -y
conda activate imagebind

pip install -r requirements.txt
```

Then you can run the app by

```python
gradio app.py
```

It should download the correct model, it might take a while.

## Seed the database

We use images from [`lexica`](https://lexica.art/) gently borrowed by this [hugging face dataset](https://huggingface.co/datasets/xfh/lexica_6k). You need to download them and edit `scripts/create_embeddings.py`. 

## Contributing

Thanks a lot for considering. The only requirements is that you run `make style` so all the code is nice and formatted :)

