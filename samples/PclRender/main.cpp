#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>


// define the point cloud type name
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cout << "point cloud display." << endl;

    // flags: 0 - gray, 1 - color, -1 - keep the raw data type
    cv::Mat color_im = cv::imread("../../data/segm/_left_184.jpg", 1);
    cv::Mat mask_im = cv::imread("../../data/segm/_left_184.png", 1);
    //cv::Mat raw_depth_map = cv::imread("../../data/result/mask/aligned_depth_map.png", -1);
    // read the raw depth data from xml file
    cv::Mat raw_depth_map;
    FileStorage fs("../../data/segm/aligned_depth_map.xml", FileStorage::READ);
    fs["Mat"] >> raw_depth_map;

    resize(color_im, color_im, raw_depth_map.size(), 0, 0, cv::INTER_NEAREST);
    resize(mask_im, mask_im, raw_depth_map.size(), 0, 0, cv::INTER_NEAREST);


    // Intrinsic (depth camera)
    const double camera_factor = 10; // use it if read from png image
    const double depth_fx = 360.4729; const double depth_fy = 361.2690;
    const double depth_u0 = 247.9301; const double depth_v0 = 205.5918;
    
    // create a point cloud (auto pointer)
    PointCloud::Ptr cloud( new PointCloud );

    for (int row_idx = 0; row_idx < raw_depth_map.rows; ++row_idx)
    {
        for (int col_idx = 0; col_idx < raw_depth_map.cols; ++col_idx)
        {
            // get depth val
            ushort depth_val = raw_depth_map.ptr<ushort>(row_idx)[col_idx];
            if (depth_val == 0) continue;

            // get mask val
            unsigned char blue = mask_im.at<cv::Vec3b>(row_idx, col_idx)[0];
            unsigned char green = mask_im.at<cv::Vec3b>(row_idx, col_idx)[1];
            unsigned char red = mask_im.at<cv::Vec3b>(row_idx, col_idx)[2];
            if ( (blue == 0) && (green == 0) && (red == 0) ) continue;

            // create a point 
            PointT point;

            //point.z = double(depth_val) / camera_factor;
            point.z = double(depth_val);
            point.x = (col_idx - depth_u0) * point.z / depth_fx;
            point.y = (row_idx - depth_v0) * point.z / depth_fy;

            point.b = color_im.at<cv::Vec3b>(row_idx, col_idx)[0];
            point.g = color_im.at<cv::Vec3b>(row_idx, col_idx)[1];
            point.r = color_im.at<cv::Vec3b>(row_idx, col_idx)[2];

            cloud->points.push_back( point );
        }
    }

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout << "point cloud size = " << cloud->points.size() << endl;

    // display pcl point cloud
    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( cloud );
    while( !viewer.wasStopped() )
    {
    }

    cloud->is_dense = false;
    //pcl::io::savePCDFile("../../data/result/mask/pointcloud.pcd", *cloud);
    
    // PLY Writer
    pcl::PLYWriter PlyWriter;
    PlyWriter.write("./test.ply", *cloud, false, true);
    PlyWriter.write("./test_binary.ply", *cloud, true, true);
    

    cloud->points.clear();
    cout << "Point cloud saved." << endl;

    return 0;
}

