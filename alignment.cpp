#include "alignment.hpp"
#include "tools.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_map>

//#define ENABLE_PERFORMANCE_METRICS

#ifdef ENABLE_PERFORMANCE_METRICS

// Performance measurement utilities
class PerformanceMetrics {
public:
    static PerformanceMetrics& getInstance() {
        static PerformanceMetrics instance;
        return instance;
    }

    void startTimer(const std::string& label) {
        timers[label] = std::chrono::high_resolution_clock::now();
    }

    double endTimer(const std::string& label) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = timers.find(label);
        if (it == timers.end()) {
            std::cerr << "Timer '" << label << "' was never started!" << std::endl;
            return 0.0;
        }
        
        double duration = std::chrono::duration<double, std::milli>(end - it->second).count();
        
        if (metrics.find(label) == metrics.end()) {
            metrics[label] = {duration, 1, duration, duration};
        } else {
            auto& metric = metrics[label];
            metric.totalTime += duration;
            metric.callCount++;
            metric.minTime = std::min(metric.minTime, duration);
            metric.maxTime = std::max(metric.maxTime, duration);
        }
        
        return duration;
    }

    void logMetric(const std::string& label, double value) {
        if (customMetrics.find(label) == customMetrics.end()) {
            customMetrics[label] = {value, 1, value, value};
        } else {
            auto& metric = customMetrics[label];
            metric.totalTime += value;
            metric.callCount++;
            metric.minTime = std::min(metric.minTime, value);
            metric.maxTime = std::max(metric.maxTime, value);
        }
    }

    void printAllMetrics() {
        std::cout << "\n==== PERFORMANCE METRICS ====\n";
        std::cout << std::left << std::setw(40) << "Function/Label" 
                  << std::setw(15) << "Avg Time (ms)" 
                  << std::setw(15) << "Total Time (ms)" 
                  << std::setw(15) << "Calls" 
                  << std::setw(15) << "Min (ms)" 
                  << std::setw(15) << "Max (ms)" 
                  << std::endl;
        std::cout << std::string(115, '-') << std::endl;

        // Print function timing metrics
        for (const auto& pair : metrics) {
            const auto& metric = pair.second;
            double avgTime = metric.totalTime / metric.callCount;
            
            std::cout << std::left << std::setw(40) << pair.first 
                      << std::fixed << std::setprecision(3)
                      << std::setw(15) << avgTime
                      << std::setw(15) << metric.totalTime
                      << std::setw(15) << metric.callCount
                      << std::setw(15) << metric.minTime 
                      << std::setw(15) << metric.maxTime
                      << std::endl;
        }

        // Print custom metrics
        if (!customMetrics.empty()) {
            std::cout << "\n==== CUSTOM METRICS ====\n";
            for (const auto& pair : customMetrics) {
                const auto& metric = pair.second;
                double avgValue = metric.totalTime / metric.callCount;
                
                std::cout << std::left << std::setw(40) << pair.first 
                          << std::fixed << std::setprecision(3)
                          << std::setw(15) << avgValue
                          << std::setw(15) << metric.totalTime
                          << std::setw(15) << metric.callCount
                          << std::setw(15) << metric.minTime 
                          << std::setw(15) << metric.maxTime
                          << std::endl;
            }
        }
    }

    void reset() {
        timers.clear();
        metrics.clear();
        customMetrics.clear();
    }

private:
    struct Metric {
        double totalTime;
        int callCount;
        double minTime;
        double maxTime;
    };

    std::unordered_map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> timers;
    std::unordered_map<std::string, Metric> metrics;
    std::unordered_map<std::string, Metric> customMetrics;

    PerformanceMetrics() = default;
    ~PerformanceMetrics() = default;
};

// RAII timer guard
class TimerGuard {
public:
    explicit TimerGuard(const std::string& label) : label(label) {
        PerformanceMetrics::getInstance().startTimer(label);
    }
    
    ~TimerGuard() {
        PerformanceMetrics::getInstance().endTimer(label);
    }
    
private:
    std::string label;
};

#define TIME_FUNCTION(label) \
    auto __timer_guard = std::make_unique<TimerGuard>(label)
#else
#define TIME_FUNCTION(label) ;
#endif

bool VideoAligner::ComputePyramid(const cv::Mat& inputFrame, const VideoAlignerParams& params) {
    TIME_FUNCTION("ComputePyramid");
    
    int width = inputFrame.cols;
    int height = inputFrame.rows;

    if (ScalePyramid[0].empty() || width != LastWidth || height != LastHeight) {
        TIME_FUNCTION("ComputePyramid_Setup");
        
        CurrFrameIndex = 0;
        PrevFrameIndex = 1;
        FramesAccumulated = 0;
        LastWidth = width;
        LastHeight = height;

        PyramidLevels = 0;
        do {
            PyramidLevels++;
            width /= 2;
            height /= 2;
        } while (width >= params.pyramid_min_width && height >= params.pyramid_min_height);

        ScalePyramid[0].clear();
        ScalePyramid[1].clear();
        ScalePyramid[0].resize(PyramidLevels);
        ScalePyramid[1].resize(PyramidLevels);

        width = inputFrame.cols;
        height = inputFrame.rows;

        KeyframeGradX.resize(PyramidLevels);
        KeyframeGradY.resize(PyramidLevels);
        KeyframeGradX[0] = Halide::Runtime::Buffer<float>(width, height);
        KeyframeGradY[0] = Halide::Runtime::Buffer<float>(width, height);

        for (int i = 1; i < PyramidLevels; i++) {
            width /= 2;
            height /= 2;
            ScalePyramid[0][i] = Halide::Runtime::Buffer<uint8_t>(width, height);
            ScalePyramid[1][i] = Halide::Runtime::Buffer<uint8_t>(width, height);
            KeyframeGradX[i] = Halide::Runtime::Buffer<float>(width, height);
            KeyframeGradY[i] = Halide::Runtime::Buffer<float>(width, height);
        }

        KeyframeArgMaxX.resize(PyramidLevels);
        KeyframeArgMaxY.resize(PyramidLevels);
        KeyframeTileSize.resize(PyramidLevels);

        KeyframeJacobianX.resize(PyramidLevels);
        KeyframeJacobianY.resize(PyramidLevels);
        SelectedPixelsX.resize(PyramidLevels);
        SelectedPixelsY.resize(PyramidLevels);
        SelectedJacobianX.resize(PyramidLevels);
        SelectedJacobianY.resize(PyramidLevels);
        WarpDiffX.resize(PyramidLevels);
        WarpDiffY.resize(PyramidLevels);
    } else {
        PrevFrameIndex = CurrFrameIndex;
        CurrFrameIndex ^= 1;
    }

    {
        TIME_FUNCTION("ConvertToBGR");
        cv::cvtColor(inputFrame, GrayInput[CurrFrameIndex], cv::COLOR_BGR2GRAY);
    }
    
    {
        TIME_FUNCTION("MatToHalide");
        ScalePyramid[CurrFrameIndex][0] = mat_to_halide_buffer_u8(GrayInput[CurrFrameIndex]);
    }

    for (int i = 1; i < PyramidLevels; i++) {
        TIME_FUNCTION("PyrDown_" + std::to_string(i));
        PyrDown(ScalePyramid[CurrFrameIndex][i-1], ScalePyramid[CurrFrameIndex][i]);
    }

    {
        TIME_FUNCTION("PhaseLayerConversion");
        cv::Mat phase_layer = halide_buffer_to_mat(ScalePyramid[CurrFrameIndex][PhaseLevel]);
        phase_layer.convertTo(PhaseImage[CurrFrameIndex], CV_32F);
    }

    if (FramesAccumulated >= 2) {
        return true;
    }
    return ++FramesAccumulated >= 2;
}

bool VideoAligner::ComputeKeyFrame() {
    TIME_FUNCTION("ComputeKeyFrame");
    
    for (int i = 0; i < PyramidLevels; i++) {
        auto& grad_x = KeyframeGradX[i];
        auto& grad_y = KeyframeGradY[i];

        {
            TIME_FUNCTION("GradXY_" + std::to_string(i));
            if (!GradXY(ScalePyramid[CurrFrameIndex][i], grad_x, grad_y)) {
                std::cerr << "Failed to compute gradient images for keyframe at level " << i << std::endl;
                return false;
            }
        }

        auto& argmax_x = KeyframeArgMaxX[i];
        auto& argmax_y = KeyframeArgMaxY[i];

        {
            TIME_FUNCTION("GradArgMax_" + std::to_string(i));
            if (!GradArgMax(grad_x, grad_y, KeyframeTileSize[i], argmax_x, argmax_y)) {
                std::cerr << "Failed to compute argmax" << std::endl;
                return false;
            }
        }

        auto& jacobian_x = KeyframeJacobianX[i];
        auto& jacobian_y = KeyframeJacobianY[i];

        {
            TIME_FUNCTION("SparseJacobian_" + std::to_string(i));
            if (!SparseJacobian(grad_x, grad_y, argmax_x, argmax_y, jacobian_x, jacobian_y)) {
                std::cerr << "Failed to compute Jacobian" << std::endl;
                return false;
            }
        }
    }

    return true;
}

static cv::Mat ComputeHessianFromSelected(
    const Halide::Runtime::Buffer<float> &selected_jacobian_x,
    const Halide::Runtime::Buffer<float> &selected_jacobian_y) {
    
    TIME_FUNCTION("ComputeHessianFromSelected");
    
    cv::Mat H = cv::Mat::zeros(4, 4, CV_64F);
    
    // Pre-compute pointers to matrix data for faster access
    double* H_data = reinterpret_cast<double*>(H.data);
    const int H_step = H.step / sizeof(double);
    
    // Process both jacobians in a single function
    auto processJacobian = [&H_data, H_step](const Halide::Runtime::Buffer<float> &jacobian) {
        const int m = jacobian.dim(0).extent();
        
        for (int i = 0; i < m; i++) {
            // Load jacobian values
            double j[4] = {
                jacobian(i, 0),
                jacobian(i, 1),
                jacobian(i, 2),
                jacobian(i, 3)
            };
            
            // Only compute upper triangular part (including diagonal)
            // due to symmetry: H(i,j) = H(j,i)
            for (int r = 0; r < 4; r++) {
                for (int c = r; c < 4; c++) {
                    H_data[r * H_step + c] += j[r] * j[c];
                }
            }
        }
    };
    
    // Process both jacobians
    {
        TIME_FUNCTION("ProcessJacobianX");
        processJacobian(selected_jacobian_x);
    }
    
    {
        TIME_FUNCTION("ProcessJacobianY");
        processJacobian(selected_jacobian_y);
    }
    
    // Copy the upper triangular values to the lower triangular part
    for (int r = 0; r < 4; r++) {
        for (int c = r + 1; c < 4; c++) {
            H_data[c * H_step + r] = H_data[r * H_step + c];
        }
    }
    
    return H;
}

bool VideoAligner::AlignNextFrame(
    const cv::Mat& inputFrame,
    SimilarityTransform& transform,
    const VideoAlignerParams& params)
{
#ifdef ENABLE_PERFORMANCE_METRICS
    TIME_FUNCTION("AlignNextFrame");
    PerformanceMetrics::getInstance().startTimer("TotalFrameTime");
#endif

    transform = SimilarityTransform(); // Identity transform

    {
        TIME_FUNCTION("ComputePyramidCall");
        if (!ComputePyramid(inputFrame, params)) {
#ifdef ENABLE_PERFORMANCE_METRICS
            double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
            PerformanceMetrics::getInstance().logMetric("IncompleteFrame", totalTime);
#endif
            return false;
        }
    }

    if (CurrFrameIndex == KeyframeIndex) {
        TIME_FUNCTION("ComputeKeyFrameCall");
        if (!ComputeKeyFrame()) {
            LastWidth = -1;
#ifdef ENABLE_PERFORMANCE_METRICS
            double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
            PerformanceMetrics::getInstance().logMetric("KeyframeFailure", totalTime);
#endif
            return false;
        }
    }

    if (params.phase_correlate) {
        TIME_FUNCTION("PhaseCorrelation");

        cv::Point2d detected_shift;
        double response = 0.0;
        detected_shift = cv::phaseCorrelate(PhaseImage[PrevFrameIndex], PhaseImage[CurrFrameIndex], cv::noArray(), &response);
#ifdef ENABLE_PERFORMANCE_METRICS
        PerformanceMetrics::getInstance().logMetric("PhaseCorrelateResponse", response);
#endif

        if (response > params.phase_correlate_threshold) {
            const float phase_layer_scale = (1 << PhaseLevel) / float(1 << PyramidLevels);
            transform.TX = detected_shift.x * phase_layer_scale;
            transform.TY = detected_shift.y * phase_layer_scale;
            if (CurrFrameIndex == KeyframeIndex) {
                transform.TX = -transform.TX;
                transform.TY = -transform.TY;
            }
        }
    }

    for (int i = PyramidLevels - 1; i >= 0; i--) {
        TIME_FUNCTION("PyramidLevel_" + std::to_string(i));
        
        int tile_size = KeyframeTileSize[i];
        auto& grad_argmax_x = KeyframeArgMaxX[i];
        auto& grad_argmax_y = KeyframeArgMaxY[i];
        auto& template_image = ScalePyramid[NonKeyframeIndex][i];
        auto& keyframe_image = ScalePyramid[KeyframeIndex][i];
        const int image_width = keyframe_image.width();
        const int image_height = keyframe_image.height();
        auto& selected_pixels_x = SelectedPixelsX[i];
        auto& selected_pixels_y = SelectedPixelsY[i];
        auto& jacobian_x = KeyframeJacobianX[i];
        auto& jacobian_y = KeyframeJacobianY[i];
        auto& selected_jacobian_x = SelectedJacobianX[i];
        auto& selected_jacobian_y = SelectedJacobianY[i];
        auto& warpdiff_x = WarpDiffX[i];
        auto& warpdiff_y = WarpDiffY[i];

        {
            TIME_FUNCTION("SparseWarpDiff_X_" + std::to_string(i));
            if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_x, transform, warpdiff_x)) {
                std::cerr << "Failed to compute warp diff at level " << i << std::endl;
#ifdef ENABLE_PERFORMANCE_METRICS
                double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("SparseWarpDiffFailure", totalTime);
#endif
                return false;
            }
        }
        
        {
            TIME_FUNCTION("SparseWarpDiff_Y_" + std::to_string(i));
            if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_y, transform, warpdiff_y)) {
                std::cerr << "Failed to compute warp diff at level " << i << std::endl;
#ifdef ENABLE_PERFORMANCE_METRICS
                double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("SparseWarpDiffFailure", totalTime);
#endif
                return false;
            }
        }

        // At the start of each pyramid level, find the argmax pixels that
        // are the closest between the template and the keyframe.
        {
            TIME_FUNCTION("DeltaPixelsSetup_" + std::to_string(i));
            
            DeltaPixelsX.clear();
            for (int j = 0; j < grad_argmax_x.height(); j++) {
                for (int k = 0; k < grad_argmax_x.width(); k++) {
                    DeltaPixel dp;
                    dp.abs_delta = warpdiff_x(k, j);
                    dp.tile_x = static_cast<uint16_t>(k);
                    dp.tile_y = static_cast<uint16_t>(j);
                    DeltaPixelsX.push_back(dp);
                }
            }
            DeltaPixelsY.clear();
            for (int j = 0; j < grad_argmax_y.height(); j++) {
                for (int k = 0; k < grad_argmax_y.width(); k++) {
                    DeltaPixel dp;
                    dp.abs_delta = warpdiff_y(k, j);
                    dp.tile_x = static_cast<uint16_t>(k);
                    dp.tile_y = static_cast<uint16_t>(j);
                    DeltaPixelsY.push_back(dp);
                }
            }
        }

        // Find the Subset of the DeltaPixels with the smallest abs_delta
        {
            TIME_FUNCTION("NthElement_" + std::to_string(i));
            
            const size_t selected_count_x =
                static_cast<size_t>(DeltaPixelsX.size() * params.smallest_fraction);
            std::nth_element(
                DeltaPixelsX.begin(),
                DeltaPixelsX.begin() + selected_count_x,
                DeltaPixelsX.end(),
                [](const DeltaPixel &lhs, const DeltaPixel &rhs) {
                    return lhs.abs_delta < rhs.abs_delta;
                }
            );
            DeltaPixelsX.resize(selected_count_x);
            
            const size_t selected_count_y =
                static_cast<size_t>(DeltaPixelsY.size() * params.smallest_fraction);
            std::nth_element(
                DeltaPixelsY.begin(),
                DeltaPixelsY.begin() + selected_count_y,
                DeltaPixelsY.end(),
                [](const DeltaPixel &lhs, const DeltaPixel &rhs) {
                    return lhs.abs_delta < rhs.abs_delta;
                }
            );
            DeltaPixelsY.resize(selected_count_y);

#ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("SelectedPointsX_" + std::to_string(i), selected_count_x);
            PerformanceMetrics::getInstance().logMetric("SelectedPointsY_" + std::to_string(i), selected_count_y);
#endif
        }

        {
            TIME_FUNCTION("BufferSetup_" + std::to_string(i));
            
            if (selected_pixels_x.dimensions() != 2 ||
                selected_pixels_x.dim(0).extent() != DeltaPixelsX.size() ||
                selected_pixels_x.dim(1).extent() != 2)
            {
                selected_pixels_x = Halide::Runtime::Buffer<uint16_t>(DeltaPixelsX.size(), 2);
            }
            if (selected_pixels_y.dimensions() != 2 ||
                selected_pixels_y.dim(0).extent() != DeltaPixelsY.size() ||
                selected_pixels_y.dim(1).extent() != 2)
            {
                selected_pixels_y = Halide::Runtime::Buffer<uint16_t>(DeltaPixelsY.size(), 2);
            }
            if (selected_jacobian_x.dimensions() != 2 ||
                selected_jacobian_x.dim(0).extent() != DeltaPixelsX.size() ||
                selected_jacobian_x.dim(1).extent() != 4)
            {
                selected_jacobian_x = Halide::Runtime::Buffer<float>(DeltaPixelsX.size(), 4);
            }
            if (selected_jacobian_y.dimensions() != 2 ||
                selected_jacobian_y.dim(0).extent() != DeltaPixelsY.size() ||
                selected_jacobian_y.dim(1).extent() != 4)
            {
                selected_jacobian_y = Halide::Runtime::Buffer<float>(DeltaPixelsY.size(), 4);
            }
        }

        {
            TIME_FUNCTION("JacobianSetup_" + std::to_string(i));
            
            for (size_t j = 0; j < DeltaPixelsX.size(); j++) {
                int tile_x = DeltaPixelsX[j].tile_x;
                int tile_y = DeltaPixelsX[j].tile_y;

                selected_pixels_x(j, 0) = grad_argmax_x(tile_x, tile_y, 0);
                selected_pixels_x(j, 1) = grad_argmax_x(tile_x, tile_y, 1);
                for (int k = 0; k < 4; k++) {
                    selected_jacobian_x(j, k) = jacobian_x(tile_x, tile_y, k);
                }
            }
            for (size_t j = 0; j < DeltaPixelsY.size(); j++) {
                int tile_x = DeltaPixelsY[j].tile_x;
                int tile_y = DeltaPixelsY[j].tile_y;

                selected_pixels_y(j, 0) = grad_argmax_y(tile_x, tile_y, 0);
                selected_pixels_y(j, 1) = grad_argmax_y(tile_x, tile_y, 1);
                for (int k = 0; k < 4; k++) {
                    selected_jacobian_y(j, k) = jacobian_y(tile_x, tile_y, k);
                }
            }
        }

        cv::Mat H;
        {
            TIME_FUNCTION("ComputeHessian_" + std::to_string(i));
            H = ComputeHessianFromSelected(selected_jacobian_x, selected_jacobian_y);
        }

        // Add condition number check using SVD
        double condition_number;
        {
            TIME_FUNCTION("SVD_" + std::to_string(i));
            cv::SVD svd(H);
            double min_sv = svd.w.at<double>(svd.w.rows-1);
            double max_sv = svd.w.at<double>(0);
            condition_number = max_sv / (min_sv + 1e-10); // Avoid division by zero

#ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("ConditionNumber_" + std::to_string(i), condition_number);
#endif

            if (condition_number > 1e6) {
                // Matrix is ill-conditioned, apply Tikhonov regularization
                double lambda = 1e-6 * max_sv;
                for (int i = 0; i < H.rows; i++) {
                    H.at<double>(i, i) += lambda;
                }
#ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().logMetric("RegularizationLambda_" + std::to_string(i), lambda);
#endif
            }
        }

        cv::Mat Hinv;
        {
            TIME_FUNCTION("MatrixInversion_" + std::to_string(i));
            Hinv = H.inv(cv::DECOMP_SVD);
        }

        int iterations_performed = 0;
        
        for (int iter = 0; iter < params.max_iters; iter++) {
            TIME_FUNCTION("ICAIteration_" + std::to_string(i) + "_" + std::to_string(iter));
            iterations_performed++;

            if (!SparseICA(
                    template_image,
                    keyframe_image,
                    selected_pixels_x,
                    selected_pixels_y,
                    selected_jacobian_x,
                    selected_jacobian_y,
                    transform,
                    IcaResult)) {
                std::cerr << "Failed to compute ICA" << std::endl;
#ifdef ENABLE_PERFORMANCE_METRICS
                double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("ICAFailure", totalTime);
#endif
                return false;
            }

            cv::Mat dt;
            {
                TIME_FUNCTION("MatrixMultiply_" + std::to_string(i) + "_" + std::to_string(iter));
                dt = Hinv * halide_vec4_to_mat(IcaResult);
            }

            // The sparse_jac kernel uses pixel coordinates normalized by image width,
            // so we need to scale the parameters by the image width.
            double scale = 1.0 / KeyframeGradX[i].width();

            SimilarityTransform delta_transform;
            delta_transform.A = dt.at<double>(0) * scale;
            delta_transform.B = dt.at<double>(1) * scale;
            delta_transform.TX = dt.at<double>(2);
            delta_transform.TY = dt.at<double>(3);

            Point ul0{0.f, 0.f};
            Point ur0{image_width - 1.f, 0.f};
            Point ll0{0.f, image_height - 1.f};
            Point lr0{image_width - 1.f, image_height - 1.f};

            ul0 = transform.warp(ul0);
            ur0 = transform.warp(ur0);
            ll0 = transform.warp(ll0);
            lr0 = transform.warp(lr0);

            {
                TIME_FUNCTION("TransformCompose_" + std::to_string(i) + "_" + std::to_string(iter));
                transform = delta_transform.compose(transform);
            }

            Point ul1{0.f, 0.f};
            Point ur1{image_width - 1.f, 0.f};
            Point ll1{0.f, image_height - 1.f};
            Point lr1{image_width - 1.f, image_height - 1.f};

            ul1 = transform.warp(ul1);
            ur1 = transform.warp(ur1);
            ll1 = transform.warp(ll1);
            lr1 = transform.warp(lr1);

            double ud = std::max(ul1.distance(ul0), ur1.distance(ur0));
            double ld = std::max(ll1.distance(ll0), lr1.distance(lr0));
            double displacement = std::max(ud, ld);
            
#ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("Displacement_" + std::to_string(i) + "_" + std::to_string(iter), displacement);
#endif

            if (displacement < params.threshold) {
                break;
            }

            if (iter >= params.max_iters - 1) {
#ifdef ENABLE_PERFORMANCE_METRICS
                double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("MaxIterationsExceeded", totalTime);
#endif
                return false;
            }
        }
        
#ifdef ENABLE_PERFORMANCE_METRICS
        PerformanceMetrics::getInstance().logMetric("IterationsPerformed_" + std::to_string(i), iterations_performed);
#endif

        if (i > 0) {
            // Move from half-resolution to full-resolution
            transform.TX *= 2.0;
            transform.TY *= 2.0;
        }
    }

    if (CurrFrameIndex != KeyframeIndex) {
        TIME_FUNCTION("TransformInverse");
        transform = transform.inverse();
    }

#ifdef ENABLE_PERFORMANCE_METRICS
    double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
    PerformanceMetrics::getInstance().logMetric("SuccessfulFrame", totalTime);
    static int frameCount = 0;
    if (++frameCount % 100 == 0) {
        PerformanceMetrics::getInstance().printAllMetrics();
    }
#endif
    
    return true;
}
