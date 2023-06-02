import gradio as gr 
import numpy as np 
from models.model_utils import get_embeddings, get_model, ModalityType
from typing import Optional
from PIL import Image
model = get_model()

def search_button_handler(text_query: Optional[str], image_query: Optional[Image.Image], audio_query: Optional[str]):
    if not text_query and not image_query and not audio_query:
        print("No inputs!")
        return
    # we have to pass a list for each query
    if text_query is not None:
        text_query = [text_query]
    if image_query is not None:
        image_query = [image_query]
    if audio_query is not None:
        audio_query = [audio_query]
    print(text_query)
    embeddings = get_embeddings(model, text_query, image_query, audio_query)
    print(embeddings[ModalityType.TEXT].shape)

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
            search_button.click(search_button_handler, [text_query, image_query, audio_query])
demo.queue()
demo.launch()