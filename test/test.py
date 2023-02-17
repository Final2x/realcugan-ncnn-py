import sys
import os
import time
from realcugan_ncnn_vulkan import Realcugan
from PIL import Image


if __name__ == "__main__":

    print("System version: ", sys.version)
    print("Test START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    _gpuid = -1

    if _gpuid == -1:
        print("USE  ~~~~~~~~~~~~~~~~~CPU~~~~~~~~~~~~~~~~~~")
    else:
        print("USE  ~~~~~~~~~~~~~~~~~GPU~~~~~~~~~~~~~~~~~~")

    time_start = time.time()
    _scale = 2

    out_w = 0
    out_h = 0

    with Image.open("test.png") as image:
        out_w = image.width * _scale
        out_h = image.height * _scale

        realcugan = Realcugan(gpuid=_gpuid, scale=_scale, noise=3)
        image = realcugan.process(image)
        image.save("output.png")

    with Image.open("output.png") as image:
        assert image.width == out_w
        assert image.height == out_h
        assert os.path.getsize("output.png") > 0

    print("Test END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Time : ", time.time() - time_start)
