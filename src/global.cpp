#include <chrono>
#include "global.h"

using namespace slic;
using namespace rbd;


const std::string kInputVideoFile = "./data/";
const std::string kInputModelFile = "./data/net/voc-8c.prototxt";
const std::string kInputWeightFile = "./data/net/voc-8c.caffemodel";
std::vector<ARectangle> rects_optimized;
std::vector<std::thread> threads_optimize;
std::mutex kOptimizeMutex;
volatile int loop = 1;


cv::Rect RectResize(ARectangle& rect, const cv::Size& img_size)
{
    int x = rect.x;
    int y = rect.y;
    int width = rect.width;
    int height = rect.height;

    // Resize (padding)
    int left = x - (width >> 2);
    int right = x + 5*(width >> 2);
    int top = y - (height >> 2);
    int btm = y + 5*(height >> 2);
    //// Not resize
    //int left = x;
    //int right = x + width;
    //int top = y;
    //int btm = y + height;

    if(left < 0) left = 0;
    if(right >= img_size.width) right = img_size.width-1;
    if(top < 0) top = 0;
    if(btm >= img_size.height) btm = img_size.height-1;

    return cv::Rect(left, top, right-left, btm-top);
}

void RbdOptimize(cv::Mat img)
{
    // Parameters Init
    SuperpixelRbd::RbdSettings rbd_settings;
    SuperPixels::Settings settings;
    settings.img_size.x = 100; // cols
    settings.img_size.y = 100; // rows
    settings.no_segs = 100; // (num-1)^2
    settings.spixel_size = 10;
    //settings.img_size.x = 416; // cols
    //settings.img_size.y = 416; // rows
    //settings.no_segs = 1000; // (num-1)^2
    //settings.spixel_size = 175;
    settings.coh_weight = 1.0f;
    settings.no_iters = 5;
    //settings.color_space = SuperPixels::ColorSpace::XYZ;
    settings.color_space = SuperPixels::ColorSpace::CIELAB;
    settings.seg_method = SuperPixels::SegMethod::GIVEN_NUM;
    settings.do_enforce_connectivity = true;
    rbd_settings.spixel_settings = settings;

    SuperpixelRbd spixel_rbd(rbd_settings);

#ifndef DETAIL_SHOW
    cv::Mat_<double> img_saliency;
    auto start = std::chrono::system_clock::now();
    spixel_rbd.Saliency(img, img_saliency, SuperpixelRbd::SaliencyType::kBackgroundProb);
    auto end = std::chrono::system_clock::now();
    std::cout << "saliency: "  << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us" << std::endl;
#else
    // 1. show superpixel segment result of ROI with bgr space
    // 2. show saliency result of ROI with lab space 
    std::lock_guard<std::mutex> guard(kOptimizeMutex);
    cv::Mat spixel_im;
    SuperPixels spixel(settings);

    //std::chrono::time_point<std::chrono::system_clock> start, end;
    auto start = std::chrono::system_clock::now();
    spixel.Process(img);
    auto end = std::chrono::system_clock::now();
    std::cout << "spixel: "  << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us" << std::endl;

    spixel.Draw(spixel_im);

    //cv::Mat_<double> img_saliency;
    //spixel_rbd.Saliency(img, img_saliency, SuperpixelRbd::SaliencyType::kBackgroundProb);

    cv::imshow("spixel", spixel_im);
    cv::imwrite("./data/result/spixel.jpg", spixel_im);
    cv::Mat resized_object_roi;
    cv::resize(img, resized_object_roi, cv::Size(100,100));
    //cv::resize(img, resized_object_roi, cv::Size(416,416));
    cv::imwrite("./data/result/resized_roi.jpg", resized_object_roi);
    char keyval = (char)cv::waitKey(0);
    if(27 == keyval) loop = 0;
#endif

    std::cout << "thread exit." << std::endl;
}
