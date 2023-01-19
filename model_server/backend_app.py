import io
import os

from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image

from model_storage import ModelStorage
from style_painter import StylePainter


app = FastAPI()
storage = ModelStorage()
style_painters = {}


@app.get('/model_list')
def list_all_models() -> dict:
    """Return a list of available models.

    :return: dict
    """
    return {"all_models": ModelStorage.available_models()}


@app.post('/transform/{style}')
def transform_image(style: str, image_data: UploadFile = File(...)) -> Response:
    """Apply style transfer to the received image. If requested `StylePainter`
    in not loaded a new object will be created.

    :param style: Name of the style to apply.
    :param image_data: Image encoded to bytes array.
    :return: Response.
    """

    # Bytes to PIL Image.
    image = Image.open(io.BytesIO(image_data.file.read()))
    # Load model if was not used before.
    if style not in style_painters:
        style_painters[style] = StylePainter(
            model_path=storage.get(style),
            device=os.getenv('SERVER_DEVICE', 'cpu')
        )

    # Apply style transfer.
    painter = style_painters[style]
    styled_image = painter.paint(image)
    # PIL Image to bytes.
    byte_array = io.BytesIO()
    styled_image.save(byte_array, format='jpeg')

    return Response(content=byte_array.getvalue(), media_type='image/jpeg')
