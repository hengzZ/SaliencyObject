#include <chrono>
#include "glance_net.h"


GlanceNet::GlanceNet(const std::string& modelFile, const std::string& weightFile, float pthreshold, float piouThreshold, float piorThreshold, int pnumClass, int pnumBox, int pgrideSize)
{
	m_modelFile = modelFile;
	m_weightFile = weightFile;
	m_threshold = pthreshold;
	m_iouThreshold = piouThreshold;
	m_iorThreshold = piorThreshold;
	m_numClass = pnumClass;
	m_numBox = pnumBox;
	m_grideSize = pgrideSize;

	// region biases
	m_region_biases = {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};

	m_boxes.resize(m_grideSize * m_grideSize * m_numBox);
	m_boxes_t.resize(m_grideSize * m_grideSize * m_numBox);
	m_cmp.probs.resize(m_grideSize * m_grideSize * m_numBox);
	m_cmp_t.probs.resize(m_grideSize * m_grideSize * m_numBox);
	for(int i=0; i < m_grideSize * m_grideSize * m_numBox; ++i)
	{
		m_cmp.probs[i].resize(m_numClass);
		m_cmp_t.probs[i].resize(m_numClass);
	}

	// GPU
	caffe::Caffe::SetDevice(0);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	m_net.reset(new caffe::Net<float>(m_modelFile, caffe::TEST));
	m_net->CopyTrainedLayersFrom(m_weightFile);

	m_inputBlob = m_net->input_blobs()[0];
	m_imgChannels = m_inputBlob->channels();
	m_imgSize = cv::Size(m_inputBlob->width(), m_inputBlob->height());
}

void GlanceNet::predict(cv::Mat& img)
{
	if(img.size() != m_imgSize){
		cv::resize(img, m_imgResized, m_imgSize);
	}
	else{
		m_imgResized = img;
	}
	m_boxes.resize(m_grideSize*m_grideSize*m_numBox);
	m_cmp.probs.resize(m_grideSize*m_grideSize*m_numBox);
	for(int i=0; i<m_grideSize * m_grideSize * m_numBox; ++i)
		m_cmp.probs[i].resize(m_numClass);
	//wrapInputBlob(m_imgResized);
	wrapInputBlobWithRgb(m_imgResized);
	//m_result = forward();
	m_result = forwardWithRegion();
	//parse(m_result, m_boxes, m_cmp.probs);
	parseRegion(m_result, m_boxes, m_cmp.probs);
	doSort(m_boxes, m_cmp.probs);
	draw(img, filter(img, m_boxes, m_cmp.probs));
}

void GlanceNet::predict(cv::Mat& img, std::vector<ARectangle>& rects)
{
	if(img.size() != m_imgSize){
		cv::resize(img, m_imgResized, m_imgSize);
	}
	else{
		m_imgResized = img;
	}
	m_boxes.resize(m_grideSize*m_grideSize*m_numBox);
	m_cmp.probs.resize(m_grideSize*m_grideSize*m_numBox);
	for(int i=0; i<m_grideSize * m_grideSize * m_numBox; ++i)
		m_cmp.probs[i].resize(m_numClass);
	//wrapInputBlob(m_imgResized);
	wrapInputBlobWithRgb(m_imgResized);
	//m_result = forward();
	m_result = forwardWithRegion();
	//parse(m_result, m_boxes, m_cmp.probs);
	parseRegion(m_result, m_boxes, m_cmp.probs);
	doSort(m_boxes, m_cmp.probs);
	
	rects = filter(img,  m_boxes, m_cmp.probs);
}

vector<float> GlanceNet::forward()
{
	vector<float> output;
	const float* blobData;

	//double start_time = aocl_utils::getCurrentTimestamp();
	m_net->Forward();
	//printf("m_net->Forward: %g ms\n", (aocl_utils::getCurrentTimestamp() - start_time) * 1000);
	boost::shared_ptr<caffe::Blob<float> > outputBlob = m_net->blob_by_name("result");
	int dims = outputBlob->count();
	blobData = outputBlob->cpu_data();
	copy(blobData, blobData+dims, back_inserter(output));
	return output;
}

vector<float> GlanceNet::forwardWithRegion()
{
	vector<float> output;
	const float* blobData;

	//double start_time = aocl_utils::getCurrentTimestamp();
	m_net->Forward();
	//printf("m_net->Forward: %g ms\n", (aocl_utils::getCurrentTimestamp() - start_time) * 1000);
	boost::shared_ptr<caffe::Blob<float> > outputBlob = m_net->blob_by_name("result");
	int dims = outputBlob->count();
	blobData = outputBlob->cpu_data();
	copy(blobData, blobData+dims, back_inserter(output));
	// Forward region operator
	region(output);
	return output;
}

void GlanceNet::wrapInputBlob(const cv::Mat& imgResized)
{
	const float scale = 0.0039215686;
	caffe::TransformationParameter param;
	param.set_scale(scale);
	caffe::DataTransformer<float> dt(param, caffe::TEST);
	dt.Transform(imgResized, m_inputBlob);
}

void GlanceNet::wrapInputBlobWithRgb(const cv::Mat& imgResized)
{
	cv::Mat img_cvt;
	cv::cvtColor(imgResized, img_cvt, cv::COLOR_BGR2RGB);

	const float scale = 0.0039215686;
	caffe::TransformationParameter param;
	param.set_scale(scale);
	caffe::DataTransformer<float> dt(param, caffe::TEST);
	dt.Transform(img_cvt, m_inputBlob);
}

void GlanceNet::parse(const vector<float>& result, vector<Box>& boxes, vector<vector<float> >& probs)
{
	// m_grideSize * m_grideSize * m_numClass + m_grideSize * m_grideSize * confidence + m_grideSize * m_grideSize * (m_numBox * 4)
	// default   7*7*3 + 7*7*2 + 7*7*2*4
	// 4 means (x, y, width, height)
	for(int i=0; i<m_grideSize*m_grideSize; ++i){
		int row = i / m_grideSize;
		int col = i % m_grideSize;
		for(int j=0; j<m_numBox; ++j){
			int index = i*m_numBox + j;
			int pIndex = m_grideSize * m_grideSize * m_numClass + index;
			int boxIndex = m_grideSize * m_grideSize * (m_numClass + m_numBox) + (index << 2);
			float scale = result[pIndex];
			boxes[index].x = (result[boxIndex + 0] + col) / m_grideSize;
			boxes[index].y = (result[boxIndex + 1] + row) / m_grideSize;
			boxes[index].w = pow(result[boxIndex + 2], 2);
			boxes[index].h = pow(result[boxIndex + 3], 2);

			for(int k=0; k<m_numClass; ++k){
				int classIndex = i*m_numClass;
				float prob = scale*result[classIndex+k];
				probs[index][k] = (prob > m_threshold) ? prob : 0;
			}
		}
	}
}

void GlanceNet::parseRegion(const vector<float>& result, vector<Box>& boxes, vector<vector<float> >& probs)
{
    // One Box Block with (4 + 1 + num_class) channels
    // note: 4 means (x, y, width, height), 1 means Pobj
    for(int i=0; i<m_grideSize*m_grideSize; ++i){
    //for(int i=85; i<86; ++i){
	int row = i / m_grideSize;
	int col = i % m_grideSize;
	for(int n=0; n<m_numBox; ++n){
	    int index = n*m_grideSize*m_grideSize + i;
	    int box_index = entryIndex(index, 0);
	    int obj_index = entryIndex(index, 4);
	    float scale = result[obj_index];
	    int stride = m_grideSize*m_grideSize;
	    boxes[index].x = (col + result[box_index + 0*stride]) / m_grideSize;
	    boxes[index].y = (row + result[box_index + 1*stride]) / m_grideSize;
	    boxes[index].w = exp(result[box_index + 2*stride]) * m_region_biases[2*n] / m_grideSize;
	    boxes[index].h = exp(result[box_index + 3*stride]) * m_region_biases[2*n+1] / m_grideSize;
	    
	    for(int k=0; k<m_numClass; ++k){
		int class_index = entryIndex(index, 5+k);
		float prob = scale * result[class_index];
		probs[index][k] = (prob > m_threshold) ? prob : 0;
	    }
	}
    }
}

void GlanceNet::doSort(vector<Box>& boxes, vector<vector<float> >& probs)
{
	int total = m_grideSize * m_grideSize * m_numBox;

	for(int i=0; i<total; ++i){
		int any_t=0;
		for(int j=0; j<m_numClass; ++j){
			any_t = any_t || probs[i][j] > 0;
		}
		if(!any_t) continue;

		Box &a = boxes[i];
		for(int j=i+1; j<total; ++j)
		{
			Box &b = boxes[j];
			if(boxIou(a,b) > m_iouThreshold){
				for(int k=0; k<m_numClass; ++k)
				{
					if(probs[i][k] < probs[j][k])
						probs[i][k] = 0.0;
					else
						probs[j][k] = 0.0;
				}
			}else if(boxIor(a,b) > m_iorThreshold){
				for(int k=0; k<m_numClass; ++k)
				{
					if(probs[i][k] < probs[j][k])
						probs[i][k] = 0.0;
					else
						probs[j][k] = 0.0;
				}
			}
		}
	}
}

vector<ARectangle> GlanceNet::filter(cv::Mat& img, vector<Box>& boxes, vector<vector<float> >& probs)
{
	vector<ARectangle> rects;
	rects.clear();
	int num = m_grideSize * m_grideSize * m_numBox;
	int wImg = img.cols;
	int hImg = img.rows;
	for(int i = 0; i < num; ++i){
		int kind = maxIndex(probs[i], m_numClass);
		float prob = probs[i][kind];
		if(prob > m_threshold){
			const Box& b = boxes[i];

			int left  = (b.x-b.w/2.)*wImg;
			int right = (b.x+b.w/2.)*wImg;
			int top   = (b.y-b.h/2.)*hImg;
			int bot   = (b.y+b.h/2.)*hImg;

			if(left < 0) left = 0;
			if(right > wImg-1) right = wImg-1;
			if(top < 0) top = 0;
			if(bot > hImg-1) bot = hImg-1;

			ARectangle rt;
			rt.x = left;
			rt.y = top;
			rt.width = right - left;
			rt.height = bot - top;
			rt.type = kind;
			rt.prob = prob;
			rects.push_back(rt);
		}
	}
	return rects;
}

void GlanceNet::draw(cv::Mat& img, vector<ARectangle> rects)
{
	static const string names[] = {"cookie", "bear", "dog", "wipe", "noodle", "milk", "puffed", "drink"};
	//cv::Mat demo = cv::imread("samples/fcn/ne/2_303.png", 1);
	//img = demo;
	for(unsigned int i = 0; i < rects.size(); ++i){
		int kind = rects[i].type;
		int w = rects[i].width;
		int h = rects[i].height;
		int x = rects[i].x + (w >> 1); 
		int y = rects[i].y + (h >> 1);
		cout << "x y w h " << x << "\t" << y << "\t" << w << "\t" << h << std::endl;
		cv::Point leftTop = cv::Point(rects[i].x, rects[i].y);
		cv::Point rightBottom = cv::Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height);
		if(0 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(255, 20 ,147), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 20 ,147));
		}
		else if (1 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(255, 0 ,0), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 0 ,0));
		}
		else if (2 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(250, 128 ,114), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(250, 128 ,114));
		}
		else if (3 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(244, 164 ,96), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(244, 164 ,96));
		}
		else if (4 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(240, 230 ,140), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(240, 230 ,140));
		}
		else if (5 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(240, 128 ,128), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(240, 128 ,128));
		}
		else if (6 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(238, 130 ,238), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(238, 130 ,238));
		}
		else if (7 == kind) {
			cv::rectangle(img, leftTop, rightBottom, cv::Scalar(50, 205 ,50), 5);
			putText(img, names[kind].c_str(), cv::Point(leftTop.x, leftTop.y - 5), CV_FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(50, 205 ,50));
		}
		cout << names[kind] << ":\t" << rects[i].prob << endl;
	}
}

// intersection-over-region
float GlanceNet::boxIor(const Box& a, const Box& b)
{
    float iora = boxIntersection(a, b) / (a.w * a.h);
    float iorb = boxIntersection(a, b) / (b.w * b.h);
    return (iora > iorb) ? iora : iorb;
}

float GlanceNet::boxIou(const Box& a, const Box& b)
{
	return boxIntersection(a, b) / boxUnion(a,b);
}

float GlanceNet::boxIntersection(const Box& a, const Box& b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if(w<0 || h<0) return 0;
	return w*h;
}

float GlanceNet::boxUnion(const Box& a, const Box& b)
{
	return a.w * a.h + b.w * b.h - boxIntersection(a, b);
}

float GlanceNet::overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1/2;
	float l2 = x2 - w2/2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1/2;
	float r2 = x2 + w2/2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

int GlanceNet::maxIndex(const vector<float>& a, int n)
{
	if(n<=0) return -1;
	int i, max_i = 0;
	float maxn = a[0];
	for(i = 1; i < n; ++i){
		// cout<<"a "<<i<<"\t"<<a[i]<<endl;
		if(a[i] > maxn){
			maxn = a[i];
			max_i = i;
		}
	}
	return max_i;
}

void GlanceNet::region(vector<float>& result)
{
	// Logistic activation for x,y and Pobj
	for(int n=0; n<m_numBox; ++n){
	    int idx = entryIndex(n*m_grideSize*m_grideSize, 0);
	    logisticArray(result, idx, 2*m_grideSize*m_grideSize);
	    idx = entryIndex(n*m_grideSize*m_grideSize, 4);
	    logisticArray(result, idx, m_grideSize*m_grideSize);
	}
	// Softmax activation for classes
	int idx = entryIndex(0, 5);
	softmaxArray(result, idx, m_numClass, m_numBox, result.size()/m_numBox, m_grideSize*m_grideSize, 1, m_grideSize*m_grideSize, 1);
}

int GlanceNet::entryIndex(int location, int entry)
{
	int n = location / (m_grideSize*m_grideSize);
	int loc = location % (m_grideSize*m_grideSize);
	return n * (m_grideSize*m_grideSize) * (4+1+m_numClass) + entry * (m_grideSize*m_grideSize) +loc;
}

void GlanceNet::logisticArray(vector<float>& result, int idx_start, const int n)
{
	for(int i=0; i<n; ++i){
	    result[idx_start+i] = 1. / ( 1. + exp(-result[idx_start+i]) );
	}
}

void GlanceNet::softmaxArray(vector<float>& result, int idx_start, int num_class, int num_box, int box_offset, int location, int loc_offset, int stride, float temp)
{
    for(int n=0; n<num_box; ++n){
	for(int loc=0; loc<location; ++loc){
	    softmaxOperator(result, idx_start+n*box_offset+loc*loc_offset, num_class, stride, temp);
	}
    }
}

void GlanceNet::softmaxOperator(vector<float>& result, int idx_start, int num_class, int stride, float temp)
{
    float sum = 0;
    float largest = -FLT_MAX;

    for(int i=0; i<num_class; ++i){
	if(result[idx_start+i*stride] > largest) largest = result[idx_start+i*stride];
    }
    for(int i=0; i<num_class; ++i){
	float e = exp(result[idx_start+i*stride]/temp - largest/temp);
	sum += e;
	result[idx_start+i*stride] = e;
    }
    for(int i=0; i<num_class; ++i){
	result[idx_start+i*stride] /= sum;
    }
}

