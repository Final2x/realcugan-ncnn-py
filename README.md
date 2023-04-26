# realcugan-ncnn-py
Python Binding for realcugan-ncnn-vulkan with PyBind11 [![PyPI version](https://badge.fury.io/py/realcugan-ncnn-py.svg?123456)](https://badge.fury.io/py/realcugan-ncnn-py?123456) [![test_pip](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/test_pip.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/test_pip.yml)  [![Release](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/Release.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/Release.yml)

RealCUGAN is a Generative Adversarial Network (GAN) based model for image super-resolution (SR). This wrapper provides an easy-to-use interface for running the pre-trained RealCUGAN model.

### Current building status matrix
| System        | Status                                                                                                                                                                                                                              | CPU (32bit)  |  CPU (64bit)       | GPU (32bit)  | GPU (64bit)        |
|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|:------------------:|:------------:|:------------------:|
| Linux (Clang) | [![CI-Linux-x64-Clang](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml)                   | —            | :white_check_mark: | —            | :white_check_mark: |
| Linux (GCC)   | [![CI-Linux-x64-GCC](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml)                         | —            | :white_check_mark: | —            | :white_check_mark: |
| Windows       | [![CI-Windows-x64-MSVC](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml)                | —            | :white_check_mark: | —            | :white_check_mark: |
| MacOS         | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) | —            | :white_check_mark: | —            | :white_check_mark: |
| MacOS (ARM)   | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) | —            | :white_check_mark: | —            | :white_check_mark: |




# Usage
```Python >= 3.6 (>= 3.9 in MacOS arm)```

To use this package, simply install it via pip:
```sh
pip install realcugan-ncnn-py
```
For Linux user:
```sh
apt install -y libomp5 libvulkan-dev
```
Then, import the Realcugan class from the package:

```python
from realcugan_ncnn_py import Realcugan
```
To initialize the model:

```python
realcugan = Realcugan(gpuid: int = 0, tta_mode: bool = False, num_threads: int = 1, noise: int = -1, scale: int = 2, tilesize: int = 0, syncgap: int = 3, model: str = "models-se", **_kwargs)
# model can be "models-se" or "models-pro" or "models-nose"
# or an absolute path to the models' directory
```
Here, gpuid specifies the GPU device to use (-1 means use CPU), tta_mode enables test-time augmentation, num_threads sets the number of threads for processing, noise specifies the level of noise to apply to the image (-1 to 3), scale is the scaling factor for super-resolution (1 to 4), tilesize specifies the tile size for processing (0 or >= 32), syncgap is the sync gap mode, and model specifies the name of the pre-trained model to use.

Once the model is initialized, you can use the upscale method to super-resolve your images:

### Pillow
```python
from PIL import Image
realcugan = Realcugan(gpuid=0, scale=2, noise=3)
with Image.open("input.jpg") as image:
    image = realcugan.process_pil(image)
    image.save("output.jpg", quality=95)
```

### opencv-python
```python
import cv2
realcugan = Realcugan(gpuid=0, scale=2, noise=3)
image = cv2.imdecode(np.fromfile("input.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
image = realcugan.process_cv2(image)
cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")
```

### ffmpeg
```python
import subprocess as sp
# your ffmpeg parameters
command_out = [FFMPEG_BIN,........] 
command_in = [FFMPEG_BIN,........]
pipe_out = sp.Popen(command_out, stdout=sp.PIPE, bufsize=10 ** 8)
pipe_in = sp.Popen(command_in, stdin=sp.PIPE)
realcugan = Realcugan(gpuid=0, scale=2, noise=3)
while True:
    raw_image = pipe_out.stdout.read(src_width * src_height * 3)
    if not raw_image:
        break
    raw_image = realcugan.process_bytes(raw_image, src_width, src_height, 3)
    pipe_in.stdin.write(raw_image)
```
# Build
[here](https://github.com/Tohrusky/realcugan-ncnn-py/blob/main/.github/workflows/Release.yml) 

*The project just only been tested in Ubuntu 18+ and Debian 9+ environments on Linux, so if the project does not work on your system, please try building it.*


# References
The following references were used in the development of this project:

[nihui/realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) - This project was the main inspiration for our work. It provided the core implementation of the Real-CUGAN algorithm using the ncnn and Vulkan libraries.

[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - Real-CUGAN is an AI super resolution model for anime images, trained in a million scale anime dataset, using the same architecture as Waifu2x-CUNet.

[media2x/realcugan-ncnn-vulkan-python](https://github.com/media2x/realcugan-ncnn-vulkan-python) - This project was used as a reference for implementing the wrapper. *Special thanks* to the original author for sharing the code. 

[ncnn](https://github.com/Tencent/ncnn) - ncnn is a high-performance neural network inference framework developed by Tencent AI Lab. 

# License
This project is licensed under the BSD 3-Clause - see the [LICENSE file](https://github.com/Tohrusky/realcugan-ncnn-py/blob/main/LICENSE) for details.
