import pathlib
from typing import List, Tuple

import requests


class ModelStorage:
    _remote_model_path = {
        'mosaic': 'mosaic-9.onnx',
        'candy': 'candy-9.onnx',
        'rain-princess': 'rain-princess-9.onnx',
        'udine': 'udnie-9.onnx',
        'pointilism': 'pointilism-9.onnx',
    }

    def __init__(
            self, models_dir: pathlib.Path = pathlib.Path('models')) -> None:
        """Class representing the model storage. It's capable of listing
        the models present in the storage and downloading requested models
        if the remote location is provided.

        :param models_dir: Path to a folder where models will be stored.
        """
        self._models_dir = models_dir
        self._url_root = (
            'https://github.com/onnx/models/raw/main/vision/'
            'style_transfer/fast_neural_style/model/'
        )

    @classmethod
    def available_models(cls) -> List[str]:
        """List all known models.

        :return: List[str]
        """
        return list(cls._remote_model_path.keys())

    def get(self, model_name: str) -> str:
        """Get a path to requested model. In case the model is not available
        locally it will download it from remote if path is provided.

        :param model_name: Model name.
        :return: Path as str.
        """
        if model_name not in self._remote_model_path:
            raise RuntimeError(
                f'model "{model_name}" not found '
                f'in {self._models_dir} nor remote.')

        if model_name not in self._models_in_dir():
            self._download_model(model_name)

        return str(self._models_dir / f'{model_name}.onnx')

    def _download_model(self, model_name: str) -> None:
        """Download model from remote.

        :param model_name: Model name.
        :return:
        """
        if not self._models_dir.exists():
            self._models_dir.mkdir(parents=True)

        url = f'{self._url_root}/{self._remote_model_path[model_name]}'
        model_path = self._models_dir / f'{model_name}.onnx'
        with model_path.open('wb') as f:
            f.write(
                requests.get(url).content
            )

    def _models_in_dir(self) -> Tuple[str]:
        """List models in `self._models_dir`"""
        return tuple(
            f.name.rstrip('.onnx')
            for f
            in self._models_dir.glob('*.onnx')
        )
