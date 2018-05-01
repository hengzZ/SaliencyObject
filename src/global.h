#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glance_net.h"
#include "slic_superpixels.h"
#include "superpixel_rbd.h"


extern const std::string kInputVideoFile;
extern const std::string kInputModelFile;
extern const std::string kInputWeightFile;
extern std::vector<ARectangle> rects_optimized;
extern std::vector<std::thread> threads_optimize;
extern volatile int loop;


cv::Rect RectResize(ARectangle& rect, const cv::Size& img_size);
void RbdOptimize(cv::Mat img);

#endif // GLOBAL_H
