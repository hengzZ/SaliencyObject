#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "global.h"
#include "glance_net.h"
#include "slic_superpixels.h"
#include "superpixel_rbd.h"
#include "load_images.h"

using namespace std;
using namespace cv;
using slic::SuperPixels;
using rbd::SuperpixelRbd;

static const std::string image_dir = "test/";
static const std::string dest_dir = "samples/fcn/result/";


int main(int argc, char** argv)
{

    //VideoCapture capt(kInputVideoFile);
    //if(!capt.isOpened()){
    //    fprintf(stderr, "error: Could not open capture.\n");
    //    return -1;
    //}
    namedWindow("demo", WINDOW_AUTOSIZE);
    std::vector<std::string> files_list;
    GetAllFiles(image_dir, files_list);

    Mat frame;
    // GlanceNet
    GlanceNet glance_net(kInputModelFile, kInputWeightFile, 0.2, 0.5, 0.5, 8, 5, 13);
    //GlanceNet glance_net(kInputModelFile, kInputWeightFile, 0.0, 1.1, 1.1, 8, 5, 13);


    //while(loop){
    //    capt >> frame;

    for(int i = 0; i < 8000 && loop; ++i){

        if((int)files_list.size() < i+1){
            cout << "no image to load... finish." << endl;
            break;
        }
        frame = cv::imread(image_dir+files_list[i]);

        if(frame.empty()){
            fprintf(stderr, "Error: encounter empty frame.\n ");
            break;
        }

        vector<ARectangle> rects;

        auto start = std::chrono::system_clock::now();
        // Glance rendering
        //glance_net.predict(frame);
        // ROI
        glance_net.predict(frame, rects);
        std::cout << ">>FILE: " << files_list[i] << std::endl;
        std::cout << "region number: " << rects.size() << std::endl;
        auto end = std::chrono::system_clock::now();
        std::cout << "predict: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us" << std::endl;

#ifdef DETAIL_SHOW
        imshow("ResizedFrame", glance_net.m_imgResized);
#endif

        rects_optimized.resize(rects.size());
        threads_optimize.resize(rects.size());
        for(int i=0; i < (int)rects.size(); ++i)
        {
            if (i<0) {
                Mat temp = frame;
                threads_optimize[0] = thread(RbdOptimize, temp);
            } else {
                cv::Rect object_roi = RectResize(rects[i], frame.size());
                cout << object_roi << endl;
                Mat temp = frame(object_roi);
                threads_optimize[i] = thread(RbdOptimize, temp);
            }
        }
        for(int i=0; i < (int)threads_optimize.size(); ++i)
        {
            threads_optimize[i].join();
        }

        imshow("demo", frame);
        //imwrite(dest_dir+files_list[i], frame);
        char keyval = (char)waitKey(0);
        if(27 == keyval) loop = 0;
    }

    destroyAllWindows();
    return 0;
}
