#ifndef REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H
#define REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H

#include "realcugan.h"
#include "pybind11/include/pybind11/pybind11.h"
#include <locale>
#include <codecvt>
#include <utility>

// wrapper class of ncnn::Mat
class RealCUGANImage {
public:
    std::string d;
    int w;
    int h;
    int c;

    RealCUGANImage(std::string d, int w, int h, int c);

    void set_data(std::string data);

    pybind11::bytes get_data() const;
};

class RealCUGANWrapped : public RealCUGAN {
public:
    RealCUGANWrapped(int gpuid, bool tta_mode, int num_threads);

    int get_tilesize(int _scale) const;

    // realcugan parameters
    void set_parameters(int _noise, int _scale, int _prepadding, int _syncgap, int _tilesize);

    int load(const std::string &parampath, const std::string &modelpath);

    int process(const RealCUGANImage &inimage, RealCUGANImage &outimage) const;

    int process_cpu(const RealCUGANImage &inimage, RealCUGANImage &outimage) const;

private:
    int gpuid;
};

int get_gpu_count();

void destroy_gpu_instance();

#endif // REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H
