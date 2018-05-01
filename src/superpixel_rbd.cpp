#include <cerrno>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <functional>
#include <mutex> // mutex
#include <opencv2/highgui/highgui.hpp> // imwrite

#include "boost/config.hpp"
#include "boost/property_map/property_map.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/graph/johnson_all_pairs_shortest.hpp"

#include "superpixel_rbd.h"
#include "fastmath.h"


namespace rbd{

static std::mutex kDisplayMutex; // mutex for displaying details

SuperpixelRbd::RbdSettings::RbdSettings()
{
    spixel_settings.img_size.x = 100; // cols
    spixel_settings.img_size.y = 100; //rows
    spixel_settings.no_segs = 100; // (num-1)^2
    spixel_settings.spixel_size = 10;
    spixel_settings.coh_weight = 1.0f;
    spixel_settings.no_iters = 5;
    spixel_settings.color_space = slic::SuperPixels::ColorSpace::XYZ;
    spixel_settings.seg_method = slic::SuperPixels::SegMethod::GIVEN_NUM;
    spixel_settings.do_enforce_connectivity = true;

    bnd_thickness = 5;
    select_boundary_sp = true;
    is_dynamic_paras = false;
    geo_sigma = 20.0;
    nei_sigma = 10.0;
    bd_con_sigma = 0.5;
    spa_sigma = 0.25;
    is_fix_high_bd_con_sp = true;
    high_thresh = 2.0;
    is_remove_low_vals = false;
    mu = 0.1;
    bg_lambda = 5.0;
}

SuperpixelRbd::SuperpixelRbd(RbdSettings& rbd_settings)
{
    rbd_settings_ = rbd_settings;
    superpixels_ = new slic::SuperPixels(rbd_settings_.spixel_settings);
}

SuperpixelRbd::~SuperpixelRbd()
{
    if(superpixels_ != NULL){
        delete superpixels_;
    }
}

// NOTE: 
// clip_val:
// geo_sigma:
// nei_sigma:
void SuperpixelRbd::Saliency(const cv::Mat_<cv::Vec3b>& image, cv::Mat_<double>& image_demo, const SaliencyType type_demo)
{
    cv::Mat_<cv::Vec3b> img_lab;
    std::vector<slic::SuperPixels::SuperPixel> superpixels;
    cv::Mat_<double> adjc_matrix;
    cv::Mat_<int> segment_result;
    std::vector<int> boundary_ids;
    cv::Mat_<double> color_distance_matrix, position_distance_matrix;
    double clip_val, geo_sigma, nei_sigma;

    cv::cvtColor(image, img_lab, cv::COLOR_BGR2Lab);
    superpixels_->GetAllSuperPixelsWithAdjcMatrixFromMat(img_lab, superpixels, adjc_matrix);
    superpixels_->ExportSegmentResultToMatWithInt(segment_result);
    int num_superpixels = superpixels.size();
    GetBoundaryPatchIds(segment_result, num_superpixels, boundary_ids);

#ifdef DETAIL_SHOW
    printf("boundary ids size: %d\n", (int)boundary_ids.size());
    for(int i=0; i < (int)boundary_ids.size(); ++i)
    {
        printf("%d\t", boundary_ids[i]);
    }
    printf("\n");
#endif

    GetColorAndPositionDistanceMatrix(superpixels, color_distance_matrix, position_distance_matrix);
    EstimateDynamicParameters(adjc_matrix, color_distance_matrix, clip_val, geo_sigma, nei_sigma);

    // Saliency Optimization
    std::vector<double> bg_prob, bg_weight,  bd_con, w_ctr, opt_w_ctr;
    EstimateBgProb(adjc_matrix, boundary_ids, color_distance_matrix, bg_prob, bd_con, bg_weight, clip_val, geo_sigma);
    CalculateWeightedContrast(color_distance_matrix, position_distance_matrix, bg_prob, w_ctr);

    SaliencyOptimization(adjc_matrix, boundary_ids, color_distance_matrix, bg_weight, w_ctr, opt_w_ctr, nei_sigma);
    // bc_con, bg_prob, bg_weight, w_ctr, opt_w_ctr
    cv::Mat_<double> saliency_img_bg = Mix(bg_prob, segment_result);
    cv::Mat_<double> saliency_img_w_ctr = Mix(w_ctr, segment_result);
    cv::Mat_<double> saliency_img_opt_w_ctr = Mix(opt_w_ctr, segment_result);

    switch(type_demo){
        case kBackgroundProb:

            break;
        case kWCtr:

            break;
        case kOptWCtr:

            break;
        default:
            break;
    }
    
#ifdef DETAIL_SHOW
    std::lock_guard<std::mutex> guard(kDisplayMutex);
    cv::imwrite("OrgROT.jpg", image);
    cv::imwrite("LabROI.jpg", img_lab);
    cv::imwrite("SegLabROI.jpg", segment_result);
    cv::Mat img_sp;
    superpixels_->Draw(img_sp);
    cv::imwrite("SpLabROI.jpg", img_sp);
    cv::imwrite("adjc_matrix.jpg", adjc_matrix);
    cv::imwrite("color_distance_matrix.jpg", color_distance_matrix);
    cv::imwrite("position_distance_matrix.jpg", position_distance_matrix);
    cv::imwrite("saliency_img_bg.jpg", saliency_img_bg);
    cv::imwrite("saliency_img_w_ctr.jpg", saliency_img_w_ctr);
    cv::imwrite("saliency_img_opt_w_ctr.jpg", saliency_img_opt_w_ctr);
#endif

}

void SuperpixelRbd::GetBoundaryPatchIds(const cv::Mat_<int>& segment_result, int num_superpixels, std::vector<int>& boundary_ids)
{
    std::vector<int> superpixels_ids(num_superpixels, 0);
    int thickness = rbd_settings_.bnd_thickness;
    for(int y = 0; y < segment_result.rows; ++y){
        for(int x = 0; x < segment_result.cols; ++x){
            if(y < thickness || y > (segment_result.rows - thickness - 1) ||
                    x < thickness || x > (segment_result.cols - thickness - 1)){
                superpixels_ids[segment_result(y,x)] = 1;
            }
        }
    }
    boundary_ids.clear();
    for(int i = 0; i < (int)superpixels_ids.size(); ++i){
        if(1 == superpixels_ids[i]){
            boundary_ids.push_back(i);
        }
    }
}

void SuperpixelRbd::GetColorAndPositionDistanceMatrix(std::vector<slic::SuperPixels::SuperPixel>& superpixels, cv::Mat_<double>& color_distance_matrix, cv::Mat_<double>& position_distance_matrix)
{
    int num_superpixels = superpixels.size();
    color_distance_matrix = cv::Mat_<double>::zeros(num_superpixels, num_superpixels);
    position_distance_matrix = cv::Mat_<double>::zeros(num_superpixels, num_superpixels);

    for(int y = 0; y < num_superpixels; ++y){
        cv::Vec3f mean_color = superpixels[y].mean_color;
        //cv::Vec2f mean_position = superpixels[y].mean_position;
        cv::Vec2f normal_mean_position = superpixels[y].normal_mean_position;
        for(int x = 0; x < num_superpixels; ++x){
            cv::Vec3f color_distance = superpixels[x].mean_color - mean_color;
            cv::Vec2f position_distance = superpixels[x].normal_mean_position - normal_mean_position;
            color_distance_matrix(y, x) = (double)sqrt(color_distance.dot(color_distance));
            position_distance_matrix(y, x) = (double)sqrt(position_distance.dot(position_distance));
        }
    }
}

void SuperpixelRbd::EstimateDynamicParameters(cv::Mat_<double>& adjc_matrix, cv::Mat_<double>& color_distance_matrix, double& clip_val, double& geo_sigma, double& nei_sigma)
{
    int num_superpixels = adjc_matrix.rows;
    
    cv::Mat_<double> adjc_matrix_diag = adjc_matrix.clone();
    for(int i = 0; i < num_superpixels; ++i){
        // Super Pixels do not link with itself for min distance analysis
        adjc_matrix_diag(i, i) = 0.0;
    }
    // Reachability matrix (between 2 layer neighbors)
    cv::Mat_<double> adjc_matrix_nn = adjc_matrix_diag * adjc_matrix_diag + adjc_matrix_diag;
    for(int i = 0; i < num_superpixels; ++i){
        // Super Pixels do not link with itself for min distance analysis
        adjc_matrix_nn(i, i) = 0.0;
    }

    std::vector<double> min_distance_1(num_superpixels, 0.0); // distance from sp_i to its neighbor
    std::vector<double> min_distance_2(num_superpixels, 0.0); // distance from sp_i to its neighbor (between 2 layer neighbors)
    std::vector<double> min_distance_3;

    for(int i = 0; i < num_superpixels; ++i){
        std::vector<double> temp_distance_1, temp_distance_2;
        for(int j = 0; j < num_superpixels; ++j){
            if(adjc_matrix_diag(i, j) > 0){
                temp_distance_1.push_back(color_distance_matrix(i, j));
            }
            if(adjc_matrix_nn(i, j) > 0){
                temp_distance_2.push_back(color_distance_matrix(i, j));
            }
            if(j < i){
                if(adjc_matrix_diag(i, j) > 0){
                    min_distance_3.push_back(color_distance_matrix(i, j));
                }
            }
        }
        if(0 == temp_distance_1.size() || 0 == temp_distance_2.size()){
            // Not reachable
            double max_color_distance = 500.0;
            min_distance_1[i] = max_color_distance;
            min_distance_2[i] = max_color_distance;
        }else{
            min_distance_1[i] = *std::min_element(temp_distance_1.begin(), temp_distance_1.end());
            min_distance_2[i] = *std::min_element(temp_distance_2.begin(), temp_distance_2.end());
        }
    }

    double sum_1 = 0.0;
    double sum_2 = 0.0;
    for(int i = 0; i < (int)min_distance_1.size(); ++i){
        sum_1 = sum_1 + min_distance_1[i];
    }
    for(int i = 0; i < (int)min_distance_2.size(); ++i){
        sum_2 = sum_2 + min_distance_2[i];
    }
    double mean_min_distance_1 = sum_1 / (double)min_distance_1.size();
    double mean_min_distance_2 = sum_2 / (double)min_distance_2.size();

    if(mean_min_distance_2 > mean_min_distance_1) {
        std::perror("error: mean_min_distance_2 should not large than mean_min_distance_1");
        assert(0);
        exit(1);
    }

    //std::sort(min_distance_3.begin(), min_distance_3.end(),[](double a, double b){return a > b;})
    std::sort(min_distance_3.begin(), min_distance_3.end(),std::greater<double>());
    double sum_3 = 0.0;
    int count = 1e-10;
    for(int i = 0; i < (int)(0.01 * min_distance_3.size()); ++i){
        sum_3 = sum_3 + min_distance_3[i];
        count += 1;
    }
    double mean_top = sum_3 / (double)count;

    clip_val = mean_min_distance_2;
    if(rbd_settings_.is_dynamic_paras){
        // TODO: Emperically choose parameters.
        geo_sigma = (3.0 * mean_min_distance_1) > 10 ? 10 : (3.0 * mean_min_distance_1);
        geo_sigma = geo_sigma > (mean_top / 10.0) ? (mean_top / 10.0) : geo_sigma;
        geo_sigma = (geo_sigma > 5) ? geo_sigma: 5;

        // TODO: Emperically choose parameters.
        nei_sigma = (3.0 * mean_min_distance_1) > (0.2 * mean_top) ? (0.2 * mean_top) : (3.0 * mean_min_distance_1);
        nei_sigma = (nei_sigma > 20) ? 20 : nei_sigma;
    }else{
        geo_sigma = rbd_settings_.geo_sigma;
        nei_sigma = rbd_settings_.nei_sigma;
    }
}

void SuperpixelRbd::EstimateBgProb(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, const cv::Mat_<double>& color_distance_matrix, std::vector<double>& bg_prob, std::vector<double>& bd_con, std::vector<double>& bg_weight, const double clip_val, const double geo_sigma)
{
    int num_superpixels = adjc_matrix.cols;
    std::vector<double> len_bnd, area;
    BoundaryConnectivity(adjc_matrix, boundary_ids, color_distance_matrix, bd_con, len_bnd, area, clip_val, geo_sigma, true);
    bg_prob = std::vector<double>(num_superpixels, 0.0);
    bg_weight = std::vector<double>(num_superpixels, 0.0);
    for(int i = 0; i < (int)bd_con.size(); ++i){
        double temp_bg_prob = 1.0 - ( fast_exp( (float)( (-0.5*bd_con[i]*bd_con[i])/(rbd_settings_.bd_con_sigma*rbd_settings_.bd_con_sigma))));
        bg_prob[i] = temp_bg_prob;
        if(rbd_settings_.is_fix_high_bd_con_sp){
            if(temp_bg_prob > rbd_settings_.high_thresh)
                bg_weight[i] = 1000.0;
            else 
                bg_weight[i] = temp_bg_prob;
        }
    }
}

void SuperpixelRbd::BoundaryConnectivity(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, const cv::Mat_<double>& color_distance_matrix, std::vector<double>& bd_con, std::vector<double>& len_bnd, std::vector<double>& area, const double clip_val, const double geo_sigma, const bool link_boundary)
{
    int num_superpixels = adjc_matrix.cols;
    cv::Mat_<double> adjc_matrix_link_bnd;
    if(link_boundary){
        LinkBoundarySps(adjc_matrix, boundary_ids, adjc_matrix_link_bnd);
    }
    cv::Mat_<double> edge_weight(color_distance_matrix.rows, color_distance_matrix.cols, 0.0);
    for(int i = 0; i < color_distance_matrix.rows; ++i){
        for(int j = 0; j < color_distance_matrix.cols; ++j){
            if(adjc_matrix_link_bnd(i, j) > 0)
                edge_weight(i, j) = (color_distance_matrix(i, j) - clip_val) > 0 ? (color_distance_matrix(i, j) - clip_val) : 0;
        }
    }
    // Calculate pair-wise shortest path cost (geodesic distance)
    cv::Mat_<double>geo_distance_matrix;
    GraphAllShortestPaths(adjc_matrix_link_bnd, edge_weight, geo_distance_matrix);
    cv::Mat_<double> w_geo;
    Dist2WeightMatrix(geo_distance_matrix, geo_sigma, w_geo);
    len_bnd = std::vector<double>(num_superpixels, 0.0);
    area = std::vector<double>(num_superpixels, 0.0);
    bd_con = std::vector<double>(num_superpixels, 0.0);
    for(int i = 0; i < w_geo.rows; ++i){
        double temp_len_bnd = 0.0;
        double temp_area = 0.0;
        for(int j = 0; j < w_geo.cols; ++j){
            if(j < (int)boundary_ids.size()){
                temp_len_bnd = temp_len_bnd + w_geo(i, boundary_ids[j]);
            }
            temp_area = temp_area + w_geo(i,j);
        }
        len_bnd[i] = temp_len_bnd;
        area[i] = temp_area;
        bd_con[i] = temp_len_bnd / sqrt(temp_area);
    }
}

void SuperpixelRbd::GraphAllShortestPaths(const cv::Mat_<double>& adjc_matrix, const cv::Mat_<double>& edge_weight_matrix, cv::Mat_<double>& geo_distance_matrix)
{
    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS, no_property, property<edge_weight_t, double, property<edge_weight2_t, double> > > Graph;
    typedef std::pair<int , int> Edge;

    int V = adjc_matrix.rows;
    std::vector<cv::Point> adjc_points;
    for(int i = 0; i < adjc_matrix.rows; ++i){
        for(int j = 0; j < adjc_matrix.cols; ++j){
            if(i > j){
                if(adjc_matrix(i, j) > 0)
                    adjc_points.push_back(cv::Point(j, i));
            }else{
                break;
            }
        }
    }

    const std::size_t E = adjc_points.size();
    Edge* pEdges = new Edge[E];
    double* pWeights = new double[E];
    for(int i = 0; i < (int)E; ++i){
        pEdges[i] = Edge(adjc_points[i].y, adjc_points[i].x);
        pWeights[i] = edge_weight_matrix(adjc_points[i].y, adjc_points[i].x);
    }
    
    Graph g(pEdges, pEdges+E, V);
    property_map<Graph, edge_weight_t>::type w = get(edge_weight, g);
    double* wp = pWeights;
    graph_traits<Graph>::edge_iterator e, e_end;
    for(boost::tie(e, e_end) = edges(g); e != e_end; ++e){
        w[*e] = *wp++;
    }
    std::vector<double>d(V, std::numeric_limits<double>::max());
    double **D = new double *[V];
    for(int i = 0; i < V; ++i){
        D[i] = new double[V];
    }
    johnson_all_pairs_shortest_paths(g, D, distance_map(&d[0]));
    geo_distance_matrix = cv::Mat_<double>(V, V, 0.0);
    for(int i = 0; i < V; ++i){
        memcpy(geo_distance_matrix .ptr<double>(i), *(D+i), V*sizeof(double));
    }
    for(int i = 0; i < V; ++i){
        delete[] D[i];
    }
    delete[] D;
    delete[] pEdges;
    delete[] pWeights;
}

void SuperpixelRbd::CalculateWeightedContrast(cv::Mat_<double>& color_distance_matrix, cv::Mat_<double>& position_distance_matrix, std::vector<double>& bg_prob, std::vector<double>& w_ctr)
{
    double eps = 1e-10;
    int num_superpixels = color_distance_matrix.cols;
    cv::Mat_<double> position_weight;
    Dist2WeightMatrix(position_distance_matrix, rbd_settings_.spa_sigma, position_weight);

    // BgProb weighted contrast
    cv::Mat_<double> w_ctr_mat;
    //cv::Mat_<double> bg_prob_mat(bg_prob);
    cv::Mat_<double> bg_prob_mat(bg_prob);
    w_ctr_mat = color_distance_matrix.mul(position_weight) * bg_prob_mat;
    // w_ctr_mat = color_distance_matrix.mul(position_weight) * cv::Mat_<double>::ones(num_superpixels, 1);
    normalize(w_ctr_mat, w_ctr_mat, 1.0, 0.0, cv::NORM_MINMAX);

    w_ctr = std::vector<double>(num_superpixels, 0.0);
    if(!rbd_settings_.is_remove_low_vals){
        for(int i = 0; i < num_superpixels; ++i){
            double temp_val = w_ctr_mat(i, 0);
            w_ctr[i] = temp_val;
        }
    }else{
        cv::Mat_<double> w_ctr_mat_0_255;
        w_ctr_mat_0_255 = 255 * w_ctr_mat;
        cv::Mat_<uchar> w_ctr_mat_uchar, temp_mat;
        w_ctr_mat_0_255.convertTo(w_ctr_mat_uchar, CV_8UC1);
        double thresh_val = cv::threshold(w_ctr_mat_uchar, temp_mat, 0, 255, cv::THRESH_OTSU);
        thresh_val = thresh_val / 255;
        for(int i = 0; i < num_superpixels; ++i){
            double temp_val = w_ctr_mat(i,0);
            if(temp_val >= thresh_val)
                w_ctr[i] = temp_val;
        }
    }
}

void SuperpixelRbd::Dist2WeightMatrix(const cv::Mat_<double>& distance_matrix, const double distance_sigma, cv::Mat_<double>& distance_weight_matrix)
{
    int num_superpixels = distance_matrix.cols;
    distance_weight_matrix = cv::Mat_<double>(distance_matrix.rows, distance_matrix.cols, 0.0);
    for(int i = 0; i < distance_weight_matrix.rows; ++i){
        for(int j = 0; j < distance_weight_matrix.cols; ++j){
            if(distance_matrix(i, j) > 3*distance_sigma)
                distance_weight_matrix(i, j) = 0.0;
            else
                distance_weight_matrix(i, j) = fast_exp( (float)(-0.5*distance_matrix(i,j)*distance_matrix(i,j)/(distance_sigma*distance_sigma)) );
        }
    }
}

void SuperpixelRbd::LinkBoundarySps(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, cv::Mat_<double>& adjc_matrix_link_bnd)
{
    adjc_matrix_link_bnd = adjc_matrix.clone();
    for(int i = 0; i < (int)boundary_ids.size(); ++i){
        int temp_y = boundary_ids[i];
        for(int j = 0; j < (int)boundary_ids.size(); ++j){
            int temp_x = boundary_ids[j];
            adjc_matrix_link_bnd(temp_y, temp_x) = 1;
        }
    }
}

void SuperpixelRbd::LinkNnAndBoundary(const cv::Mat_<double>& adjc_matrix, const std::vector<int>& boundary_ids, cv::Mat_<double>& adjc_matrix_nn)
{
    int num_superpixels = adjc_matrix.cols;

    adjc_matrix_nn = cv::Mat_<double>(num_superpixels, num_superpixels, 0.0);
    cv::Mat_<double> temp_adjc_matrix = adjc_matrix * adjc_matrix + adjc_matrix;
    //cv::Mat_<double> temp_adjc_matrix = adjc_matrix.clone();
    for(int i = 0; i < adjc_matrix_nn.rows; ++i){
        for(int j = 0; j < adjc_matrix_nn.cols; ++j){
            if(temp_adjc_matrix(i, j) > 0)
                adjc_matrix_nn(i, j) = 1.0;
        }
    }
    for(int i = 0; i < (int)boundary_ids.size(); ++i){
        int temp_y = boundary_ids[i];
        for(int j = 0; j < (int)boundary_ids.size(); ++j){
            int temp_x = boundary_ids[j];
            adjc_matrix_nn(temp_y, temp_x) = 1.0;
        }
    }
}

void SuperpixelRbd::SaliencyOptimization(cv::Mat_<double>& adjc_matrix, std::vector<int>& boundary_ids, cv::Mat_<double>& color_distance_matrix, std::vector<double>& bg_weight, std::vector<double>& fg_weight, std::vector<double>& opt_w_ctr, double nei_sigma)
{
    int num_superpixels = adjc_matrix.cols;
    // Link nn boundary
    cv::Mat_<double> adjc_matrix_nn;
    LinkNnAndBoundary(adjc_matrix, boundary_ids, adjc_matrix_nn);

    cv::Mat_<double> temp_color_distance_matrix = color_distance_matrix.clone();
    for(int i = 0; i < temp_color_distance_matrix.rows; ++i){
        for(int j = 0; j < temp_color_distance_matrix.cols; ++j){
            if(0 == adjc_matrix_nn(i,j))
                temp_color_distance_matrix(i,j) = 4*nei_sigma;
        }
    }
    cv::Mat_<double> Wn;
    Dist2WeightMatrix(temp_color_distance_matrix, nei_sigma, Wn);
    cv::Mat_<double> W = Wn + (rbd_settings_.mu * adjc_matrix);

    std::vector<double> sum_W(num_superpixels, 0.0);
    for(int j = 0; j < W.cols; ++j){
        double temp_val = 0;
        for(int i = 0; i < W.rows; ++i){
            temp_val += W(i,j);
        }
        sum_W[j] = temp_val;
    }

    cv::Mat_<double> D(num_superpixels, num_superpixels, 0.0);
    cv::Mat_<double> E_bg(num_superpixels, num_superpixels, 0.0);
    cv::Mat_<double> E_fg(num_superpixels, num_superpixels, 0.0);
    for(int i = 0; i < num_superpixels; ++i){
        for(int j = 0; j < num_superpixels; ++j){
            if(i >= j){
                if(i == j){
                    D(i,j) = sum_W[i];
                    E_bg(i,j) = bg_weight[i] * rbd_settings_.bg_lambda;
                    E_fg(i,j) = fg_weight[i];
                }
            }else{
                break;
            }
        }
    }
    cv::Mat_<double> opt_w_ctr_mat = (D - W + E_bg + E_fg).inv() * (E_fg * cv::Mat_<double>::ones(num_superpixels, 1));
    
    opt_w_ctr = std::vector<double>(opt_w_ctr_mat.rows, 0.0);
    for(int i = 0; i < opt_w_ctr_mat.rows; ++i){
        opt_w_ctr[i] = opt_w_ctr_mat(i,0);
    }
}

template<typename T>
cv::Mat_<T> SuperpixelRbd::Mix(const std::vector<T>& pixel_saliency, const cv::Mat_<int>& segment_result)
{
    cv::Mat_<T> saliency_img(segment_result.size());
    for(int i=0; i < segment_result.rows; ++i){
        for(int j=0; j < segment_result.cols; ++j){
            saliency_img(j,i) = pixel_saliency[segment_result(j,i)];
        }
    }
    return saliency_img;
}


} // namespace rbd
