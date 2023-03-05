#include "realcugan_wrapped.h"

// Image Data Structure
Image::Image(std::string d, int w, int h, int c) {
    this->d = std::move(d);
    this->w = w;
    this->h = h;
    this->c = c;
}

void Image::set_data(std::string data) {
    this->d = std::move(data);
}

pybind11::bytes Image::get_data() const {
    return pybind11::bytes(this->d);
}

// RealCUGANWrapped
RealCUGANWrapped::RealCUGANWrapped(int gpuid, bool tta_mode, int num_threads)
        : RealCUGAN(gpuid, tta_mode, num_threads) {
    this->gpuid = gpuid;
}

int RealCUGANWrapped::get_tilesize(int _scale) const {
    int tilesize = 0;
    if (this->gpuid == -1) return 400;

    uint32_t heap_budget = ncnn::get_gpu_device(this->gpuid)->get_heap_budget();

    if (scale == 2) {
        if (heap_budget > 1300)
            tilesize = 400;
        else if (heap_budget > 800)
            tilesize = 300;
        else if (heap_budget > 400)
            tilesize = 200;
        else if (heap_budget > 200)
            tilesize = 100;
        else
            tilesize = 32;
    }
    if (scale == 3) {
        if (heap_budget > 3300)
            tilesize = 400;
        else if (heap_budget > 1900)
            tilesize = 300;
        else if (heap_budget > 950)
            tilesize = 200;
        else if (heap_budget > 320)
            tilesize = 100;
        else
            tilesize = 32;
    }
    if (scale == 4) {
        if (heap_budget > 1690)
            tilesize = 400;
        else if (heap_budget > 980)
            tilesize = 300;
        else if (heap_budget > 530)
            tilesize = 200;
        else if (heap_budget > 240)
            tilesize = 100;
        else
            tilesize = 32;
    }

    return tilesize;
}

void RealCUGANWrapped::set_parameters(int _noise, int _scale, int _prepadding, int _syncgap, int _tilesize) {
    this->noise = _noise;
    this->scale = _scale;
    this->tilesize = _tilesize ? _tilesize : RealCUGANWrapped::get_tilesize(_scale);
    this->prepadding = _prepadding;
    this->syncgap = _syncgap;
}

int RealCUGANWrapped::load(const std::string &parampath,
                           const std::string &modelpath) {
#if _WIN32
    // convert string to wstring
    auto to_wide_string = [&](const std::string& input) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(input);
    };
    return RealCUGAN::load(to_wide_string(parampath), to_wide_string(modelpath));
#else
    return RealCUGAN::load(parampath, modelpath);
#endif
}

int RealCUGANWrapped::process(const Image &inimage, Image &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return RealCUGAN::process(inimagemat, outimagemat);
}

int RealCUGANWrapped::process_cpu(const Image &inimage, Image &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return RealCUGAN::process_cpu(inimagemat, outimagemat);
}

// ?
int get_gpu_count() { return ncnn::get_gpu_count(); }

void destroy_gpu_instance() { ncnn::destroy_gpu_instance(); }

PYBIND11_MODULE(realcugan_ncnn_vulkan_wrapper, m) {
    pybind11::class_<RealCUGANWrapped>(m, "RealCUGANWrapped")
            .def(pybind11::init<int, bool, int>())
            .def("load", &RealCUGANWrapped::load)
            .def("process", &RealCUGANWrapped::process)
            .def("process_cpu", &RealCUGANWrapped::process_cpu)
            .def("set_parameters", &RealCUGANWrapped::set_parameters);

    pybind11::class_<Image>(m, "Image")
            .def(pybind11::init<std::string, int, int, int>())
            .def("get_data", &Image::get_data)
            .def("set_data", &Image::set_data);

    m.def("get_gpu_count", &get_gpu_count);

    m.def("destroy_gpu_instance", &destroy_gpu_instance);
}
