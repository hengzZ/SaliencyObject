#ifndef GLANCE_NET_H
#define GLANCE_NET_H

#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/memory_data_layer.hpp"

using namespace std;


struct Box {
    float x,y,w,h;
};

struct Node {
    int index,kind;
};

struct ARectangle {
    int x;
    int y;
    int width;
    int height;
    int type;
    float prob;
};

class GlanceNet{
    public:
	GlanceNet(const std::string& modelFile, const std::string& weightFile, float pthreshold=0.2, float piouThreshold=0.5, float piorThreshold=0.5, int pnumClass=5, int pnumBox=5, int pgrideSize=13);
	void predict(cv::Mat& img);
	void predict(cv::Mat& img, std::vector<ARectangle>& rects);
	void wrapInputBlob(const cv::Mat& imgResized);
	void wrapInputBlobWithRgb(const cv::Mat& imgResized);

	private:
		vector<float> forward();
		vector<float> forwardWithRegion();
		void parse(const vector<float>& result, vector<Box>& boxes, vector<vector<float> >& probs);
		void parseRegion(const vector<float>& result, vector<Box>& boxes, vector<vector<float> >& probs);
		void doSort(vector<Box>& boxes, vector<vector<float> >& probs);
		vector<ARectangle> filter(cv::Mat& img, vector<Box>& boxes, vector<vector<float> >& probs);
		void draw(cv::Mat& img, vector<ARectangle> rects);
		static float boxIou(const Box& a, const Box& b);
		static float boxIor(const Box& a, const Box& b);
		static float boxIntersection(const Box& a, const Box& b);
		static float boxUnion(const Box& a, const Box& b);
		static float overlap(float x1, float w1, float x2, float w2);
		static int maxIndex(const vector<float>& a, int n);
		void region(vector<float>& result);
		int entryIndex(int location, int entry);
		void logisticArray(vector<float>& result, int idx_start, const int n);
		void softmaxArray(vector<float>& result, int idx_start, int num_class, int num_box, int box_offset, int location, int loc_offset, int stride, float temp);
		void softmaxOperator(vector<float>& result, int idx_start, int num_class, int stride, float temp);

    private:
        boost::shared_ptr<caffe::Net<float> > m_net;
        caffe::Blob<float>* m_inputBlob;
        cv::Size m_imgSize;
        int m_imgChannels;
	std::string m_modelFile;
	std::string m_weightFile;
        float m_threshold;
        float m_iouThreshold;
        float m_iorThreshold;
        int m_numClass;
        int m_numBox;
        int m_grideSize;
        struct Cmp{
            vector<vector<float> > probs;
            bool operator()(const Node& a, const Node& b){
                return probs[a.index][a.kind] > probs[b.index][b.kind];
            }
        }m_cmp, m_cmp_t;
        vector<Box> m_boxes;
        vector<Box> m_boxes_t;
        vector<float> m_result;
	vector<float> m_region_biases;

    public:
        cv::Mat m_imgResized;
};

#endif // GLANCE_NET_H

