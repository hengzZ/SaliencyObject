#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// Helper function for Kinect depth render
#define USHORT_MAX                      (0xFFFF)
#define UCHAR_MAX                       (0xFF)
#define MIN_DEPTH                       (400)       // minimum reliable depth value of Kinect
#define MAX_DEPTH                       (16383)     // maximum reliable depth value of Kinect
#define UNKNOWN_DEPTH                   (0)
#define UNKNOWN_DEPTH_COLOR             (0x00000000)
#define NEAREST_COLOR                   (0x00000000)
#define TOO_FAR_COLOR                   (0x000000FF)
#define MINIMUM(a,b)                    ((a) < (b) ? (a) : (b))


// color(gray) table for depth map rendering. (log-scaled)
unsigned char depth_color_table[USHORT_MAX + 1];
unsigned char GetIntensity(int depth);
// init color table
int InitializeDepthColorTable(void);
// rendering depth map
int RenderDepthMap(const cv::Mat& src_depth, cv::Mat& dst_depth);
int FillDepthHoles(const cv::Mat& src_depth, cv::Mat& filled_depth);
int AlignDepthWithColor(const cv::Mat& src_depth, const cv::Mat& src_color, cv::Mat& aligned_depth);


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cout << "Introdution: render the depth map for display, fill the depth map and align the depth map with color image." << endl;

    string color_im_file = "../../data/segm/_left_184.jpg";
    string depth_map_file = "../../data/segm/_raw_depth_184.xml";
    
    // xml reader
    FileStorage fs(depth_map_file, FileStorage::READ);
    // initialize the color table
    InitializeDepthColorTable();
    
    // get the raw depth map and the original color image
    Mat raw_depth_map;
    fs["Mat"] >> raw_depth_map;
    Mat color_im = imread(color_im_file, 1);

    // resize the color image to the same size
    Mat color_im_resized;
    resize(color_im, color_im_resized, raw_depth_map.size(), 0, 0, cv::INTER_NEAREST);


    // Align
    Mat aligned_depth_map;
    AlignDepthWithColor(raw_depth_map, color_im_resized, aligned_depth_map);

    Mat aligned_depth_map_disp;
    RenderDepthMap(aligned_depth_map, aligned_depth_map_disp);

    // save the aligned depth map into xml file
    imwrite("../../data/segm/aligned_depth_map.png", aligned_depth_map);
    imwrite("../../data/segm/aligned_depth_map_disp.png", aligned_depth_map_disp);
    FileStorage fw("../../data/segm/aligned_depth_map.xml", FileStorage::WRITE);
    fw << "Mat" << aligned_depth_map;
    fw.release();


    imshow("disp", aligned_depth_map_disp);
    char keyVal = waitKey(0);

    destroyAllWindows();
    return 0;
}


// Implements of Helper Functions

int InitializeDepthColorTable(void)
{
	memset(depth_color_table, 0, USHORT_MAX + 1);
	// set color for unkown depth
	depth_color_table[UNKNOWN_DEPTH] = UNKNOWN_DEPTH_COLOR;

	unsigned short min_reliable_depth = MIN_DEPTH;
	unsigned short max_reliable_depth = MAX_DEPTH;

	for (int depth = UNKNOWN_DEPTH + 1; depth < min_reliable_depth; ++depth)
	{
		depth_color_table[depth] = NEAREST_COLOR;
	}

	for (unsigned short depth = min_reliable_depth; depth <= max_reliable_depth; ++depth)
	{
		unsigned char intensity = GetIntensity(depth);
		depth_color_table[depth] = 255 - intensity;
	}

	return 0;
}

unsigned char GetIntensity(int depth)
{
	// validate the depth reliability
	if (depth < MIN_DEPTH || depth > MAX_DEPTH) return UCHAR_MAX;

	// Use a logarithmic scale that shows more detail for nearer depths.
	// The constants in this formula were chosen such that values between
	// MIN_DEPTH and MAX_DEPTH will map to the full range of possbile byte values.
	//const float depth_range_scale = 500.0f;
	//const int intensity_range_scale = 512;
	const float depth_range_scale = 10000.0f;
	const int intensity_range_scale = 1536;
	return (unsigned char)(~(unsigned char) MINIMUM(UCHAR_MAX, 
		log((double)(depth-MIN_DEPTH) / depth_range_scale + 1) * intensity_range_scale));

}

int RenderDepthMap(const cv::Mat& src_depth, cv::Mat& dst_depth)
{
    dst_depth = cv::Mat::zeros(src_depth.size(), CV_8UC1);

    for (int row_idx = 0; row_idx < src_depth.rows; ++row_idx)
    {
	uchar* ptr_dst_depth = dst_depth.ptr<uchar>(row_idx);
	const ushort* ptr_src_depth = src_depth.ptr<ushort>(row_idx);

	for ( int col_idx = 0; col_idx < src_depth.cols; ++col_idx)
	{
	    ushort depth_val = ptr_src_depth[col_idx];
	    ptr_dst_depth[col_idx] = depth_color_table[depth_val] & 0x000000FF;
	}
    }

    return 0;
}

int FillDepthHoles(const cv::Mat& src_depth, cv::Mat& filled_depth)
{
	src_depth.copyTo(filled_depth);
	double lambda = 0.25;
	double k = 25.0;

	int col_offset = 1;
	int row_offset = 3;

	for (int col = col_offset; col < src_depth.cols; ++col) 
	{
		for (int row = row_offset; row < src_depth.rows; ++row) 
		{
			// centered around the up up pixel
			ushort* ptr_filled_depth = filled_depth.ptr<ushort>(row);
			ushort* ptr_temp_up_three = filled_depth.ptr<ushort>(row - 3);
			ushort* ptr_temp_up_two = filled_depth.ptr<ushort>(row - 2);
			ushort* ptr_temp_up_one = filled_depth.ptr<ushort>(row -1);

			if (ptr_filled_depth[col] < 10) {
				double ni = static_cast<double>(ptr_temp_up_three[col] - ptr_temp_up_two[col]);
				double si = static_cast<double>(ptr_temp_up_one[col] - ptr_temp_up_two[col]);
				double wi = static_cast<double>(ptr_temp_up_two[col - 1] - ptr_temp_up_two[col]);
				double ei = static_cast<double>(ptr_temp_up_two[col + 1] - ptr_temp_up_two[col]);

				double cn = exp(-(ni * ni) / (k * k));
				double cs = exp(-(si * si) / (k * k));
				double cw = exp(-(wi * wi) / (k * k));
				double ce = exp(-(ei * ei) / (k * k));

				ptr_filled_depth[col] = static_cast<ushort>( ptr_temp_up_two[col] + lambda * (cn * ni + cs * si + cw * wi + ce * ei) + 0.5);
			}
		}
	}

	return 0;
}

int AlignDepthWithColor(const cv::Mat& src_depth, const cv::Mat& src_color, cv::Mat& aligned_depth)
{
	//// initialize the intrinsic and extrinsic parameters of the depth sensor of Kinect and color camera
	// Intrinsic (color)
	static double color_fx = 279.4825; static double color_fy = 412.5026;
	static double color_u0 = 245.3808; static double color_v0 = 222.5997;
	// Intrinsic (depth)
	static double depth_fx = 360.4729; static double depth_fy = 361.2690;
	static double depth_u0 = 247.9301; static double depth_v0 = 205.5918;
	// Extrinsic
	static cv::Mat rotation_matrix = (cv::Mat_<double>(3, 3) << 0.9999, 0.0106, -0.0015, -0.0106, 0.9999, 0.0052, 0.0015, -0.0052, 1.0000);
	static cv::Mat translation_matrix = (cv::Mat_<double>(3, 1) << 52.3004, 0.0856, -0.0617);

	int depth_map_height = src_color.rows;
	int depth_map_width = src_color.cols;

	// aligned depth map
	cv::Mat temp_depth_for_color = cv::Mat::zeros(depth_map_height, depth_map_width, CV_16UC1);

	for (int row_idx_depth = 0; row_idx_depth < depth_map_height; ++row_idx_depth)
	{
		const ushort* ptr_raw_depth_map = src_depth.ptr<ushort>(row_idx_depth);

		for (int col_idx_depth = 0; col_idx_depth < depth_map_width; ++col_idx_depth)
		{
			double raw_depth_val = static_cast<double>(ptr_raw_depth_map[col_idx_depth]);
			const double z_depth = raw_depth_val;
			const double x_depth = (col_idx_depth - depth_u0) * z_depth / depth_fx;
			const double y_depth = (row_idx_depth - depth_v0) * z_depth / depth_fy;

			cv::Mat xyz_depth = (cv::Mat_<double>(3, 1) << x_depth, y_depth, z_depth);
			cv::Mat xyz_color = rotation_matrix * xyz_depth + translation_matrix;

			const double z_color = xyz_color.at<double>(2, 0);
			const double y_color = xyz_color.at<double>(1, 0);
			const double x_color = xyz_color.at<double>(0, 0);

			double temp_col_idx_color = (x_color / z_color) * color_fx + color_u0;
			double temp_row_idx_color = (y_color / z_color) * color_fy + color_v0;

			int col_idx_color = static_cast<int>(temp_col_idx_color + 0.5);
			int row_idx_color = static_cast<int>(temp_row_idx_color + 0.5);
			if (col_idx_color < 0) col_idx_color = 0;
			if (col_idx_color >= depth_map_width) col_idx_color = (depth_map_width - 1);
			if (row_idx_color < 0) row_idx_color = 0;
			if (row_idx_color >= depth_map_height) row_idx_color = (depth_map_height - 1);

			// obtain depth value for color
			ushort* ptr_temp_depth_for_color = temp_depth_for_color.ptr<ushort>(row_idx_color);
			ptr_temp_depth_for_color[col_idx_color] = z_color;
		}
	}

	// dilate
	cv::Mat temp_dilated_depth;
	cv::Mat elemet = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(temp_depth_for_color, temp_dilated_depth, elemet);

	// fill the depth map
	FillDepthHoles(temp_dilated_depth, aligned_depth);

	// render
	cv::Mat raw_depth_disp;
	RenderDepthMap(src_depth, raw_depth_disp);

	cv::Mat aligned_depth_disp;
	RenderDepthMap(temp_depth_for_color, aligned_depth_disp);

	cv::Mat aligned_depth_with_filled_disp;
	RenderDepthMap(aligned_depth, aligned_depth_with_filled_disp);


	cv::imwrite("../../data/result/align/color.jpg", src_color);
	cv::imwrite("../../data/result/align/raw_depth_disp.jpg", raw_depth_disp);
	cv::imwrite("../../data/result/align/aligned_depth_disp.jpg", aligned_depth_disp);
	cv::imwrite("../../data/result/align/aligned_depth_with_filled_disp.jpg", aligned_depth_with_filled_disp);

	cvtColor(raw_depth_disp, raw_depth_disp, COLOR_GRAY2BGR);
	cvtColor(aligned_depth_disp, aligned_depth_disp, COLOR_GRAY2BGR);

	cv::Mat color_with_raw_depth, color_with_aligned_depth;
	addWeighted(src_color, 0.5, raw_depth_disp, 0.5, 0, color_with_raw_depth);
	addWeighted(src_color, 0.5, aligned_depth_disp, 0.5, 0, color_with_aligned_depth);

	cv::imwrite("../../data/result/align/color_with_raw_depth.jpg", color_with_raw_depth);
	cv::imwrite("../../data/result/align/color_with_aligned_depth.jpg", color_with_aligned_depth);

	return 0;
}

