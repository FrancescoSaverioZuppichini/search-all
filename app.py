import gradio as gr 
import numpy as np 
from models.model_utils import get_embeddings, get_model, ModalityType
from typing import Optional
from PIL import Image
from time import perf_counter

from dotenv import load_dotenv

load_dotenv()
from vector_store import VectorStore

vs = VectorStore.from_env()
model = get_model()

def search_button_handler(text_query: Optional[str], image_query: Optional[Image.Image], audio_query: Optional[str]):
    if not text_query and not image_query and not audio_query:
        print("No inputs!")
        return
    # we have to pass a list for each query
    print(f"text_query = {text_query}")
    if text_query == '' and len(text_query) <= 0:
        text_query = None
    if text_query is not None:
        text_query = [text_query]
    if image_query is not None:
        image_query = [image_query]
    if audio_query is not None:
        audio_query = [audio_query]
    print(text_query)
    start = perf_counter()
    embeddings = get_embeddings(model, text_query, image_query, audio_query)
    embeddings[ModalityType.TEXT] *= 0.5
    embedding = sum(embeddings.values())[0].cpu().numpy() 
    print(f"Took {(perf_counter() - start) * 1000:.2f}")
    query = f"select * from (select metadata, images, cosine_similarity(embeddings, ARRAY{embedding.tolist()}) as score from \"{vs.dataset_path}\") order by score desc limit 15"
    query_res = vs._ds.query(query, runtime = {"tensor_db": True})
    print(query_res.summary())
    images = query_res.images.data(aslist = True)['value']
    return images

with gr.Blocks() as demo:
    text_query = gr.Text(label="Text")
    with gr.Row():
        image_query = gr.Image(label="Image", type="pil")
        with gr.Column():
            audio_query = gr.Audio(label="Audio", source="microphone", type="filepath")
            search_button = gr.Button("Search", label="Search", variant="primary")
            with gr.Accordion("Settings", open=False):
                limit = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="search limit",
                    interactive=True,
                )
    gallery = gr.Gallery().style(columns=[3], object_fit="contain", height="auto")
    search_button.click(search_button_handler, [text_query, image_query, audio_query], [gallery])

demo.queue()
demo.launch()