#include <cstdio>
#include "slic_superpixels.h"


namespace slic{

SuperPixels::SuperPixel::SuperPixel()
{

}

SuperPixels::SuperPixels(const int img_width, const int img_height, const int num_segments, const int size_superpixles, const float coh_weight, const int num_iters, const ColorSpace color_space, const SegMethod seg_method, const bool do_enforce_connectivity)
{
    settings_.img_size.x = img_width;
    settings_.img_size.y = img_height;
    settings_.no_segs = num_segments;
    settings_.spixel_size = size_superpixles;
    settings_.coh_weight = coh_weight;
    settings_.no_iters = num_iters;
    settings_.color_space = color_space;
    settings_.seg_method = seg_method;
    settings_.do_enforce_connectivity = do_enforce_connectivity;

    engine_ = new Engine(settings_);

    input_img_ = new Uchar4Image(settings_.img_size, true, true);
    output_img_ = new Uchar4Image(settings_.img_size, true, true);
    img_size_ = cv::Size(img_width, img_height);
}

SuperPixels::SuperPixels(Settings& settings)
{
    settings_ = settings;
    engine_ = new Engine(settings);
    input_img_ = new Uchar4Image(settings.img_size, true, true);
    output_img_ = new Uchar4Image(settings.img_size, true, true);
    img_size_ = cv::Size(settings.img_size.x, settings.img_size.y);
}

SuperPixels::~SuperPixels()
{
    if(engine_ != NULL){
        delete engine_;
    }
    if(input_img_ != NULL){
        delete input_img_;
    }
    if(output_img_ != NULL){
        delete output_img_;
    }
}

void SuperPixels::Process(const cv::Mat& image)
{
    LoadImageFromMat(image);
    engine_->Process_Frame(input_img_);
    GetSegmentResult();
}

void SuperPixels::Draw(cv::Mat& image)
{
    engine_->Draw_Segmentation_Result(output_img_);
    ExportImageToMat(image);
}

void SuperPixels::LoadImageFromMat(const cv::Mat& in_img)
{
    cv::Mat img_resized;
    cv::resize(in_img, img_resized, img_size_);
    Vector4u* input_img_ptr = input_img_->GetData(MEMORYDEVICE_CPU);
    for(int y = 0; y < input_img_->noDims.y; ++y){
        for(int x = 0; x < input_img_->noDims.x; ++x){
            int idx = x + y * input_img_->noDims.x;
            input_img_ptr[idx].b = img_resized.at<cv::Vec3b>(y,x)[0];
            input_img_ptr[idx].g = img_resized.at<cv::Vec3b>(y,x)[1];
            input_img_ptr[idx].r = img_resized.at<cv::Vec3b>(y,x)[2];
        }
    }
}

void SuperPixels::ExportImageToMat(cv::Mat& out_img)
{
    out_img.create(img_size_, CV_8UC3);
    const Vector4u* output_img_ptr = output_img_->GetData(MEMORYDEVICE_CPU);
    for(int y = 0; y < output_img_->noDims.y; ++y){
        for(int x = 0; x < output_img_->noDims.x; ++x){
            int idx = x + y * output_img_->noDims.x;
            out_img.at<cv::Vec3b>(y,x)[0] = output_img_ptr[idx].b;
            out_img.at<cv::Vec3b>(y,x)[1] = output_img_ptr[idx].g;
            out_img.at<cv::Vec3b>(y,x)[2] = output_img_ptr[idx].r;
        }
    }
}

void SuperPixels::ExportSegmentResultToMatWithInt(cv::Mat_<int>& result)
{
    result.create(img_size_);
    const int* result_img_ptr = result_img_->GetData(MEMORYDEVICE_CPU);
    for(int y = 0; y < result_img_->noDims.y; ++y){
        for(int x = 0; x < result_img_->noDims.x; ++x){
            int idx = x + y * result_img_->noDims.x;
            result.at<int>(y,x) = result_img_ptr[idx];
        }
    }
}

int SuperPixels::GetMaxLabel()
{
    int max_label = 0;
    const int* result_img_ptr = result_img_->GetData(MEMORYDEVICE_CPU);
    for(int y=0; y < result_img_->noDims.y; ++y){
        for(int x=0; x< result_img_->noDims.x; ++x){
            int idx = x + y * result_img_->noDims.x;
            if(result_img_ptr[idx] > max_label)
                max_label = result_img_ptr[idx];
        }
    }
    return max_label;
}

void SuperPixels::GetAllSuperPixelsFromMat(const cv::Mat& image, std::vector<SuperPixel>& superpixels)
{
    Process(image);
    int num_labels = GetMaxLabel() + 1;
    //printf("superpixel size: %d\n", num_labels);

    superpixels.resize(num_labels);
    std::vector<double> count(num_labels, 1e-10); // NOTE: some cluster maybe disappear, count=0 will encounter error.
    Vector4u* input_img_ptr = input_img_->GetData(MEMORYDEVICE_CPU);
    const int* result_img_ptr = result_img_->GetData(MEMORYDEVICE_CPU);

    for(int y = 0; y < result_img_->noDims.y; ++y){
        for(int x = 0; x < result_img_->noDims.x; ++x){
            int idx = x + y * result_img_->noDims.x;
            int label = result_img_ptr[idx];
            superpixels[label].mean_color[0] += input_img_ptr[idx].b;
            superpixels[label].mean_color[1] += input_img_ptr[idx].g;
            superpixels[label].mean_color[2] += input_img_ptr[idx].r;
            superpixels[label].mean_position += cv::Vec2f(x, y);
            count[label] += 1;
        }
    }
    for(int i=0; i < num_labels; ++i){
        superpixels[i].mean_color *=  1.0 / count[i];
        superpixels[i].mean_position *= 1.0 / count[i];
        superpixels[i].size = count[i];
    }
    for(int i=0; i < num_labels; ++i){
        superpixels[i].normal_mean_color = superpixels[i].mean_color * 1.0 / 255;
        superpixels[i].normal_mean_position[0] = superpixels[i].mean_position[0] * 1.0 / (result_img_->noDims.x);
        superpixels[i].normal_mean_position[1] = superpixels[i].mean_position[1] * 1.0 / (result_img_->noDims.y);
    }
}

void SuperPixels::GetAllSuperPixelsWithAdjcMatrixFromMat(const cv::Mat& image, std::vector<SuperPixel>& superpixels, cv::Mat_<double>& adjc_matrix)
{
    Process(image);
    int num_labels = GetMaxLabel() + 1;

    superpixels.resize(num_labels);
    std::vector<double> count(num_labels, 1e-10); // NOTE: some cluster maybe disappear, count=0 will encounter error.
    Vector4u* input_img_ptr = input_img_->GetData(MEMORYDEVICE_CPU);
    const int* result_img_ptr = result_img_->GetData(MEMORYDEVICE_CPU);

    // SuperPixel
    for(int y = 0; y < result_img_->noDims.y; ++y){ // rows
        for(int x = 0; x < result_img_->noDims.x; ++x){ // cols
            int idx = x + y * result_img_->noDims.x;
            int label = result_img_ptr[idx];
            superpixels[label].mean_color[0] += input_img_ptr[idx].b;
            superpixels[label].mean_color[1] += input_img_ptr[idx].g;
            superpixels[label].mean_color[2] += input_img_ptr[idx].r;
            superpixels[label].mean_position += cv::Vec2f(x, y);
            count[label] += 1;
        }
    }
    for(int i=0; i < num_labels; ++i){
        superpixels[i].mean_color *=  1.0 / count[i];
        superpixels[i].mean_position *= 1.0 / count[i];
        superpixels[i].size = count[i];
    }
    for(int i=0; i < num_labels; ++i){
        superpixels[i].normal_mean_color = superpixels[i].mean_color * 1.0 / 255;
        superpixels[i].normal_mean_position[0] = superpixels[i].mean_position[0] * 1.0 / (result_img_->noDims.x);
        superpixels[i].normal_mean_position[1] = superpixels[i].mean_position[1] * 1.0 / (result_img_->noDims.y);
    }

#ifdef DETAIL_SHOW
    printf("superpixel size: %d\n", num_labels);
    printf("count[i]:\n");
    for(int i=0; i < (int)count.size(); ++i){
        printf("%.3f\t", count[i]);
    }
    printf("\n");
    for(int i = 0; i < (int)superpixels.size(); ++i)
    {
        if(superpixels[i].size < 5){
            printf("mean color: (%.3f, %.3f, %.3f)\n", superpixels[i].mean_color[0], superpixels[i].mean_color[1], superpixels[i].mean_color[2]);
            printf("mean position: (%.3f, %.3f)\n", superpixels[i].mean_position[0], superpixels[i].mean_position[1]);
        }
    }
#endif

    // Adjacent Matrix (4-neighbor)
    adjc_matrix = cv::Mat_<double>::eye(num_labels, num_labels);
    cv::Mat_<double> adjc_matrix_x(num_labels, num_labels, 0.0), adjc_matrix_y(num_labels, num_labels, 0.0);

    for(int y=1; y < result_img_->noDims.y; ++y){
        for(int x=1; x < result_img_->noDims.x; ++x){
            int idx = x + y * result_img_->noDims.x;
            int idx_pre_row = x + (y-1) * result_img_->noDims.x;
            int idx_pre_col = (x-1) + y * result_img_->noDims.x;

            int top_bottom_diff = result_img_ptr[idx] - result_img_ptr[idx_pre_row];
            if(0 != top_bottom_diff){
                adjc_matrix_y(result_img_ptr[idx], result_img_ptr[idx_pre_row]) = 1;
                adjc_matrix_y(result_img_ptr[idx_pre_row], result_img_ptr[idx]) = 1;
            }

            int left_right_diff = result_img_ptr[idx] - result_img_ptr[idx_pre_col];
            if(0 != left_right_diff){
                adjc_matrix_x(result_img_ptr[idx], result_img_ptr[idx_pre_col]) = 1;
                adjc_matrix_x(result_img_ptr[idx_pre_col], result_img_ptr[idx]) = 1;
            }

        }
    }

    adjc_matrix = adjc_matrix + adjc_matrix_y + adjc_matrix_x;
}

void SuperPixels::GetSegmentResult()
{
    result_img_ =  engine_->Get_Seg_Res();
}

void SuperPixels::GetSettings(Settings& settings)
{
    settings = settings_;
}

} // namespace slic
