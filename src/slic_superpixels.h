#ifndef SLIC_SUPERPIXELS_H
#define SLIC_SUPERPIXELS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gSLICr_Lib/gSLICr.h"


namespace slic{

class SuperPixels{
public:
    typedef gSLICr::engines::core_engine Engine;
    typedef gSLICr::objects::settings Settings;
    typedef gSLICr::COLOR_SPACE ColorSpace;
    typedef gSLICr::SEG_METHOD SegMethod;
    typedef gSLICr::UChar4Image Uchar4Image;
    typedef gSLICr::IntImage IntImage;
    typedef gSLICr::Vector4u Vector4u;

    struct SuperPixel{
        cv::Vec3f mean_color;
        cv::Vec3f normal_mean_color;
        cv::Vec2f mean_position;
        cv::Vec2f normal_mean_position;
        int size;
        SuperPixel();
    };

public:
    SuperPixels(const int img_width, const int img_height, const int num_segments, const int size_superpixles, const float coh_weight, const int num_iters, const ColorSpace color_space, const SegMethod seg_method, const bool do_enforce_connectivity);
    SuperPixels(Settings& settings);
    ~SuperPixels();
    void Process(const cv::Mat& image);
    void Draw(cv::Mat& image);
    void ExportSegmentResultToMatWithInt(cv::Mat_<int>& result);
    void GetAllSuperPixelsFromMat(const cv::Mat& image, std::vector<SuperPixel>& superpixels);
    // AdjcMatrix: adjacent matrix of superpixels
    void GetAllSuperPixelsWithAdjcMatrixFromMat(const cv::Mat& image, std::vector<SuperPixel>& superpixels, cv::Mat_<double>& adjcMatrix);
    void GetSettings(Settings& settings);

private:
    void LoadImageFromMat(const cv::Mat& image);
    void ExportImageToMat(cv::Mat& image);
    void GetSegmentResult();
    int GetMaxLabel();

private:
    Settings settings_;
    Engine* engine_;
    Uchar4Image* input_img_;
    Uchar4Image* output_img_;
    const IntImage* result_img_;
    cv::Size img_size_;
}; // SuperPixels


} // namespace slic

#endif // SLIC_SUPERPIXELS_H
