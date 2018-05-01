#ifndef SUPERPIXEL_RBD_H
#define SUPERPIXEL_RBD_H

#include "slic_superpixels.h"


namespace rbd{

class SuperpixelRbd{
public:
    struct RbdSettings{
        slic::SuperPixels::Settings spixel_settings;
        int bnd_thickness;
        bool select_boundary_sp;
        bool is_dynamic_paras;
        double geo_sigma;
        double nei_sigma;
        double bd_con_sigma;
        double spa_sigma;
        bool is_fix_high_bd_con_sp;
        double high_thresh;
        bool is_remove_low_vals;
        double mu;
        double bg_lambda;
        RbdSettings();
    };

    enum SaliencyType{
        kBackgroundProb,
        kWCtr,
        kOptWCtr
    };

public:
    SuperpixelRbd(RbdSettings& rbd_settings);
    ~SuperpixelRbd();
    void Saliency(const cv::Mat_<cv::Vec3b>& image, cv::Mat_<double>& image_demo, const SaliencyType type_demo);

private:

    void GetBoundaryPatchIds(const cv::Mat_<int>& segment_result, int num_superpixels, std::vector<int>& boundary_ids);
    void GetColorAndPositionDistanceMatrix(std::vector<slic::SuperPixels::SuperPixel>& superpixels, cv::Mat_<double>& color_distance_matrix, cv::Mat_<double>& position_distance_matrix);
    void EstimateDynamicParameters(cv::Mat_<double>& adjc_matrix, cv::Mat_<double>& color_distance_matrix, double& clip_val, double& geo_sigma, double& nei_sigma);
    void EstimateBgProb(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, const cv::Mat_<double>& color_distance_matrix, std::vector<double>& bg_prob, std::vector<double>& bd_con, std::vector<double>& bg_weight, const double clip_val, const double geo_sigma);
    void BoundaryConnectivity(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, const cv::Mat_<double>& color_distance_matrix, std::vector<double>& bd_con, std::vector<double>& len_bnd, std::vector<double>& area, const double clip_val, const double geo_sigma, const bool link_boundary);
    void GraphAllShortestPaths(const cv::Mat_<double>& adjc_matrix, const cv::Mat_<double>& edge_weight_matrix, cv::Mat_<double>& geo_distance_matrix);
    void CalculateWeightedContrast(cv::Mat_<double>& color_distance_matrix, cv::Mat_<double>& position_distance_matrix, std::vector<double>& bg_prob, std::vector<double>& w_ctr);
    void SaliencyOptimization(cv::Mat_<double>& adjc_matrix, std::vector<int>& boundary_ids, cv::Mat_<double>& color_distance_matrix, std::vector<double>& bg_weight, std::vector<double>& fg_weight, std::vector<double>& opt_w_ctr, double nei_sigma);
    void Dist2WeightMatrix(const cv::Mat_<double>& distance_matrix, const double distance_sigma, cv::Mat_<double>& distance_weight_matrix);
    void LinkBoundarySps(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, cv::Mat_<double>& adjc_matrix_link_bnd);
    void LinkNnAndBoundary(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, cv::Mat_<double>& adjc_matrix_nn);
    template<typename T>
    cv::Mat_<T> Mix(const std::vector<T>& pixel_saliency, const cv::Mat_<int>& segment_result);


private:
    RbdSettings rbd_settings_;
    slic::SuperPixels* superpixels_;

}; // SuperpixelRbd

} // namespace rbd

#endif // SUPERPIXEL_RBD_H
