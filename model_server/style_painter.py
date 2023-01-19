from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image


class StylePainter:

    def __init__(
            self,
            model_path: str,
            scale_factor: float = 1/2,
            device: str = 'cuda'
    ) -> None:
        """StylePainter keeps a loaded style transfer model and is capable
        of processing images to apply the style transfer. Pretrained models
        operate on 224x224 images hence post-processing upscaling is possible.
        `scale_factor` option is available to request output image scale.

        :param model_path: Path to '.onnx' file with a model.
        :param scale_factor: Scale of the output image relative to input size.
        :param device: 'CPU' or 'CUDA'.
        """
        self._scale_factor = scale_factor
        self._session = ort.InferenceSession(
            model_path,
            providers=self._get_providers(device)
        )
        self._model_output_name = self._session.get_outputs()[0].name
        self._model_input_name = self._session.get_inputs()[0].name

    def paint(self, image: Image) -> Image:
        """Pass the image through the style transfer model.

        :param image: Input `Image`.
        :return: `Image`.
        """
        width, height = image.size
        model_input = self._preprocess_input(image)
        model_output = self._run_model(model_input)
        result_image = self._postprocess_output(model_output, width, height)
        return result_image

    @classmethod
    def _get_providers(cls, device: str) -> List[str]:
        # Get providers for a requested device.
        return [p for p in ort.get_available_providers() if device.upper() in p]

    @classmethod
    def _preprocess_input(cls, image: Image) -> np.ndarray:
        # Transform `Image` into model input array.
        image_data = np.array(
            image.resize((224, 224), Image.Resampling.LANCZOS)
        ).astype('float32')
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, axis=0)
        return image_data

    def _postprocess_output(
            self, output: np.ndarray, original_width: int, original_height: int,
    ) -> Image:
        # Transform output array into an `Image`.
        result = np.clip(output, 0, 255)
        result = result.transpose(1, 2, 0).astype("uint8")
        output_size = (
            int(original_width * self._scale_factor),
            int(original_height * self._scale_factor),

        )
        result = Image.fromarray(result).resize(
            output_size, Image.Resampling.LANCZOS,
        )
        return result

    def _run_model(self, model_input: np.ndarray) -> np.ndarray:
        return self._session.run(
            [self._model_output_name],
            {self._model_input_name: model_input}
        )[0][0]
