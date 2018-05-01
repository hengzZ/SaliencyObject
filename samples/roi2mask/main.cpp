#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    cout << "Function: roi2mask - create a label mask from the object rois." << endl;

    // Color image
    Mat color_im = imread("../../data/result/left_68.jpg", 1);

    // Object ROI (rbd results)
    Mat object1 = imread("../../data/result/100pixels/left_69_1_wCtr_Optimized.png", 0);
    Mat object2 = imread("../../data/result/100pixels/left_69_2_wCtr_Optimized.png", 0);
    //Mat object3 = imread("../../data/result/100pixels/1_1_wCtr_Optimized.png", 0);

    // cv::Rect(left, top, width, height);
    Rect Rect1 = cv::Rect(545, 156, 270, 570);
    Rect Rect2 = cv::Rect(740, 63, 624, 840);
    //Rect Rect3 = cv::Rect(954, 631, 648, 448);


    // roi2mask
    threshold(object1, object1, 150, 255, cv::THRESH_BINARY);
    threshold(object2, object2, 150, 255, cv::THRESH_BINARY);
    //threshold(object3, object3, 150, 255, cv::THRESH_BINARY);

    imwrite("../../data/result/mask/object1_org.jpg", object1);
    imwrite("../../data/result/mask/object2_org.jpg", object2);
    //imwrite("../../data/result/mask/object3_org.jpg", object3);
    
    Mat mask_color = cv::Mat::zeros(color_im.size(), CV_8UC3);

    Mat roi1_mask = mask_color(Rect1);
    Mat roi2_mask = mask_color(Rect2);
    //Mat roi3_mask = mask_color(Rect3);
    resize(object1, object1, roi1_mask.size(), 0, 0, cv::INTER_NEAREST);
    resize(object2, object2, roi2_mask.size(), 0, 0, cv::INTER_NEAREST);
    //resize(object3, object3, roi3_mask.size(), 0, 0, cv::INTER_NEAREST);

    roi1_mask.setTo(cv::Scalar(255,0,0), object1);
    roi2_mask.setTo(cv::Scalar(0,255,0), object2);
    //roi3_mask.setTo(cv::Scalar(0,0,255), object3);


    // display
    imshow("color", color_im);
    imshow("disp1", object1);
    imshow("disp2", object2);
    //imshow("disp3", object3);
    imshow("mask", mask_color);
    imwrite("../../data/result/mask/object1.jpg", object1);
    imwrite("../../data/result/mask/object2.jpg", object2);
    //imwrite("../../data/result/mask/object3.jpg", object3);
    imwrite("../../data/result/mask/mask_color.jpg", mask_color);
    //imwrite("../../data/result/mask/mask_for_pcl.jpg", mask_color);
    int keyVal = waitKey(0);

    return 0;
}
