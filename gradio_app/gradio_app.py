import io
import os
from typing import List

import gradio as gr
from PIL import Image
import requests

# Gradio app settings.
GRADIO_APP_PORT = os.getenv('GRADIO_APP_PORT', 7860)
# Model server settings.
MODEL_SERVER_HOST = os.getenv('MODEL_SERVER_HOST', 'localhost')
MODEL_SERVER_PORT = os.getenv('MODEL_SERVER_PORT', 8000)
MODEL_ENDPOINT = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"


def inference(image: Image, style: str) -> Image:
    """Calls model endpoint to apply requested style to an image.

    :param image: The input image.
    :param style: Style name to apply.
    :return: `Image`
    """

    # Encoding image into a bytes array.
    # Source: https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpeg')
    file = {
        'image_data':
            ('image.jpg', image_bytes.getvalue(), 'multipart/form-data')
    }
    response = requests.post(
        f'{MODEL_ENDPOINT}/transform/{style}',
        files=file,
    )
    return Image.open(io.BytesIO(response.content))


def build_interface(style_options: List[str]) -> gr.Interface:
    """Build Grad.io app interface.

    :param style_options: List of available models.
    :return: `gr.Interface`
    """
    interface = gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Image(type="pil"),
            gr.components.Radio(
                style_options,
                type="value",
                value=style_options[0],
                label='style')
        ],
        outputs=gr.components.Image(type="pil"),
        title="Fast Style Transfer",
    )

    return interface


if __name__ == "__main__":
    available_models = requests.get(
        f'{MODEL_ENDPOINT}/model_list'
    ).json()['all_models']

    styling_app = build_interface(available_models)

    styling_app.launch(server_name="0.0.0.0", server_port=int(GRADIO_APP_PORT))
