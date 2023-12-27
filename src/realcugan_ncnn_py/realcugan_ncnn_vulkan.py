"""
The MIT License (MIT)

Copyright (c) 2021 ArchieMeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# 参考https://github.com/media2x/realcugan-ncnn-vulkan-python, 感谢原作者

import pathlib
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    from . import realcugan_ncnn_vulkan_wrapper as wrapped
except ImportError:
    import realcugan_ncnn_vulkan_wrapper as wrapped


class Realcugan:
    def __init__(
        self,
        gpuid: int = 0,
        tta_mode: bool = False,
        num_threads: int = 1,
        noise: int = -1,
        scale: int = 2,
        tilesize: int = 0,
        syncgap: int = 3,
        model: str = "models-se",
    ):
        """
        RealCUGAN class for Super Resolution

        :param gpuid: gpu device to use, -1 for cpu
        :param tta_mode: enable test time argumentation
        :param num_threads: processing thread count
        :param noise: denoise level, -1 for conservative, 0 for no denoise, 1-3 for denoise
        :param scale: upscale ratio
        :param tilesize: tile size, 0 for auto, must >= 32
        :param syncgap: sync gap mode
        :param model: realcugan model name, default is "models-se", can be "models-se" or "models-pro" or "models-nose", or an absolute path to the models' directory
        """

        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert noise in range(-1, 4), "noise must be -1-3"
        assert scale in range(1, 5), "scale must be 1-4"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert syncgap in range(4), "syncgap must be 0-3"
        assert num_threads >= 1, "num_threads must be a positive integer"

        self._gpuid = gpuid

        self._realcugan_object = wrapped.RealCUGANWrapped(gpuid, tta_mode, num_threads)

        self._model = model
        self._noise = noise
        self._scale = scale

        self._set_parameters(noise, scale, syncgap, tilesize)

        self._load()

        self.raw_in_image = None
        self.raw_out_image = None

    def _set_parameters(self, noise: int, scale: int, syncgap: int, tilesize: int) -> None:
        """
        Set parameters for RealCUGAN

        :param noise: denoise level
        :param scale: upscale ratio
        :param tilesize: tile size
        :param syncgap: sync gap mode
        :return: None
        """
        _prepadding = {2: 18, 3: 14, 4: 19}.get(self._scale, 0)
        self._realcugan_object.set_parameters(noise, scale, _prepadding, syncgap, tilesize)

    def _load(self, param_path: Optional[pathlib.Path] = None, model_path: Optional[pathlib.Path] = None) -> None:
        """
        Load models from given paths. Use self._model if one or all of the parameters are not given.

        :param param_path: the path to model params. usually ended with ".param"
        :param model_path: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if param_path is None or model_path is None:
            model_path = pathlib.Path(self._model)
            if not model_path.is_dir():
                model_path = pathlib.Path(__file__).parent / "models" / self._model

            if self._noise == -1:
                param_path = model_path / f"up{self._scale}x-conservative.param"
                model_path = model_path / f"up{self._scale}x-conservative.bin"
            elif self._noise == 0:
                param_path = model_path / f"up{self._scale}x-no-denoise.param"
                model_path = model_path / f"up{self._scale}x-no-denoise.bin"
            else:
                param_path = model_path / f"up{self._scale}x-denoise{self._noise}x.param"
                model_path = model_path / f"up{self._scale}x-denoise{self._noise}x.bin"

        if self._realcugan_object.load(str(param_path), str(model_path)) != 0:
            raise Exception("Failed to load model")

    def process(self) -> None:
        if self._gpuid != -1:
            self._realcugan_object.process(self.raw_in_image, self.raw_out_image)
        else:
            self._realcugan_object.process_cpu(self.raw_in_image, self.raw_out_image)

    def process_pil(self, _image: Image) -> Image:
        """
        Process a PIL image

        :param _image: PIL image
        :return: processed PIL image
        """

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.width * _image.height))
        out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.RealCUGANImage(in_bytes, _image.width, _image.height, channels)

        self.raw_out_image = wrapped.RealCUGANImage(
            out_bytes,
            self._scale * _image.width,
            self._scale * _image.height,
            channels,
        )

        self.process()

        return Image.frombytes(
            _image.mode,
            (
                self._scale * _image.width,
                self._scale * _image.height,
            ),
            self.raw_out_image.get_data(),
        )

    def process_cv2(self, _image: np.ndarray) -> np.ndarray:
        """
        Process a cv2 image

        :param _image: cv2 image
        :return: processed cv2 image
        """
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.shape[1] * _image.shape[0]))
        out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.RealCUGANImage(in_bytes, _image.shape[1], _image.shape[0], channels)

        self.raw_out_image = wrapped.RealCUGANImage(
            out_bytes,
            self._scale * _image.shape[1],
            self._scale * _image.shape[0],
            channels,
        )

        self.process()

        res = np.frombuffer(self.raw_out_image.get_data(), dtype=np.uint8).reshape(
            self._scale * _image.shape[0], self._scale * _image.shape[1], channels
        )

        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    def process_bytes(self, _image_bytes: bytes, width: int, height: int, channels: int) -> bytes:
        """
        Process a bytes image, like bytes from ffmpeg

        :param _image_bytes: bytes
        :param width: image width
        :param height: image height
        :param channels: image channels
        :return: processed bytes image
        """
        if self.raw_in_image is None and self.raw_out_image is None:
            self.raw_in_image = wrapped.RealCUGANImage(_image_bytes, width, height, channels)

            self.raw_out_image = wrapped.RealCUGANImage(
                (self._scale**2) * len(_image_bytes) * b"\x00",
                self._scale * width,
                self._scale * height,
                channels,
            )

        self.raw_in_image.set_data(_image_bytes)

        self.process()

        return self.raw_out_image.get_data()
