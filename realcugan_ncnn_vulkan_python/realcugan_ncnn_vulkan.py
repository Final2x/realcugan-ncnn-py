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
# 参考https://github.com/media2x/realcugan-ncnn-vulkan-python，感谢原作者

import pathlib
import time
from PIL import Image

if __package__ or "." in __name__:
    from . import realcugan_ncnn_vulkan_wrapper as wrapped
else:
    import realcugan_ncnn_vulkan_wrapper as wrapped


class Realcugan:
    """
    :param gpuid int: gpu device to use (-1=cpu)
    :param tta_mode bool: enable test time argumentation
    :param num_threads int: processing thread count
    :param noise int: denoise level
    :param scale int: upscale ratio
    :param tilesize int: tile size
    :param syncgap int: sync gap mode
    :param model str: realcugan model name
    """

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
            **_kwargs,
    ):
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

        self._set_parameters(noise, scale, self._get_prepadding(), syncgap)

        self._load()

    def _set_parameters(self, noise, scale, prepadding, syncgap) -> None:
        """
        Set parameters for RealCUGAN

        :param noise: denoise level
        :param scale: upscale ratio
        :param tilesize: tile size
        :param prepadding: prepadding
        :param syncgap: sync gap mode
        :return: None
        """
        self._realcugan_object.set_parameters(noise, scale, prepadding, syncgap)

    def _get_prepadding(self) -> int:
        if self._model in ("models-se", "models-nose", "models-pro"):
            return {2: 18, 3: 14, 4: 19}.get(self._scale, 0)
        else:
            raise ValueError(f'model "{self._model}" is not supported')

    def _load(
            self, param_path: pathlib.Path = None, model_path: pathlib.Path = None
    ) -> None:
        """
        Load models from given paths. Use self.model if one or all of the parameters are not given.

        :param parampath: the path to model params. usually ended with ".param"
        :param modelpath: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if param_path is None or model_path is None:
            model_path = pathlib.Path(self._model)
            if not model_path.is_dir():
                model_path = pathlib.Path(__file__).parent / "models" / self._model

                if self._noise == -1:
                    param_path = (
                            model_path
                            / f"up{self._scale}x-conservative.param"
                    )
                    model_path = (
                            model_path
                            / f"up{self._scale}x-conservative.bin"
                    )
                elif self._noise == 0:
                    param_path = (
                            model_path
                            / f"up{self._scale}x-no-denoise.param"
                    )
                    model_path = (
                            model_path / f"up{self._scale}x-no-denoise.bin"
                    )
                else:
                    param_path = (
                            model_path
                            / f"up{self._scale}x-denoise{self._noise}x.param"
                    )
                    model_path = (
                            model_path
                            / f"up{self._scale}x-denoise{self._noise}x.bin"
                    )
        if self._realcugan_object.load(str(param_path), str(model_path)) != 0:
            raise Exception("Failed to load model")

    def process(self, image: Image) -> Image:

        in_bytes = image.tobytes()
        channels = int(len(in_bytes) / (image.width * image.height))
        out_bytes = (self._scale ** 2) * len(in_bytes) * b"\x00"

        raw_in_image = wrapped.Image(
            in_bytes,
            image.width,
            image.height,
            channels
        )

        raw_out_image = wrapped.Image(
            out_bytes,
            self._scale * image.width,
            self._scale * image.height,
            channels,
        )

        if self._gpuid != -1:
            self._realcugan_object.process(raw_in_image, raw_out_image)
        else:
            self._realcugan_object.process_cpu(raw_in_image, raw_out_image)

        return Image.frombytes(
            image.mode,
            (
                self._scale * image.width,
                self._scale * image.height,
            ),
            raw_out_image.get_data(),
        )


if __name__ == "__main__":
    time_start = time.time()

    with Image.open("input.jpg") as image:
        realcugan = Realcugan(gpuid=0, scale=2, noise=3)
        image = realcugan.process(image)
        image.save("output.jpg")

    print(f"Time: {(time.time() - time_start) * 1000} ms")
