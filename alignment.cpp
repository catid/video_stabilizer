#include "alignment.hpp"

#include "tools.hpp"

bool VideoAligner::ComputePyramid(const cv::Mat& inputFrame) {
    int width = inputFrame.cols;
    int height = inputFrame.rows;

    if (ScalePyramid[0].empty() || width != LastWidth || height != LastHeight) {
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
        } while (width >= 20 && height >= 20);

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

    cv::cvtColor(inputFrame, GrayInput[CurrFrameIndex], cv::COLOR_BGR2GRAY);
    ScalePyramid[CurrFrameIndex][0] = mat_to_halide_buffer_u8(GrayInput[CurrFrameIndex]);

    for (int i = 1; i < PyramidLevels; i++) {
        PyrDown(ScalePyramid[CurrFrameIndex][i-1], ScalePyramid[CurrFrameIndex][i]);
    }

    cv::Mat phase_layer = halide_buffer_to_mat(ScalePyramid[CurrFrameIndex][PhaseLevel]);
    phase_layer.convertTo(PhaseImage[CurrFrameIndex], CV_32F);

    if (FramesAccumulated >= 2) {
        return true;
    }
    return ++FramesAccumulated >= 2;
}

bool VideoAligner::ComputeKeyFrame() {
    for (int i = 0; i < PyramidLevels; i++) {
        auto& grad_x = KeyframeGradX[i];
        auto& grad_y = KeyframeGradY[i];

        if (!GradXY(ScalePyramid[CurrFrameIndex][i], grad_x, grad_y)) {
            std::cerr << "Failed to compute gradient images for keyframe at level " << i << std::endl;
            return false;
        }

        auto& argmax_x = KeyframeArgMaxX[i];
        auto& argmax_y = KeyframeArgMaxY[i];

        if (!GradArgMax(grad_x, grad_y, KeyframeTileSize[i], argmax_x, argmax_y)) {
            std::cerr << "Failed to compute argmax" << std::endl;
            return false;
        }

        auto& jacobian_x = KeyframeJacobianX[i];
        auto& jacobian_y = KeyframeJacobianY[i];

        if (!SparseJacobian(grad_x, grad_y, argmax_x, argmax_y, jacobian_x, jacobian_y)) {
            std::cerr << "Failed to compute Jacobian" << std::endl;
            return false;
        }
    }

    return true;
}

static cv::Mat ComputeHessianFromSelected(
    const Halide::Runtime::Buffer<float> &selected_jacobian_x,
    const Halide::Runtime::Buffer<float> &selected_jacobian_y)
{
    const int m = selected_jacobian_x.dim(0).extent();
    cv::Mat H = cv::Mat::zeros(4, 4, CV_64F);

    for (int i = 0; i < m; i++)
    {
        // J in [4], each is float
        double j0 = selected_jacobian_x(i, 0);
        double j1 = selected_jacobian_x(i, 1);
        double j2 = selected_jacobian_x(i, 2);
        double j3 = selected_jacobian_x(i, 3);

        // Outer product Jᵀ*J, accumulate into H
        // H is symmetrical, but we’ll just fill all entries for clarity:
        H.at<double>(0,0) += j0*j0;  H.at<double>(0,1) += j0*j1;
        H.at<double>(0,2) += j0*j2;  H.at<double>(0,3) += j0*j3;

        H.at<double>(1,0) += j1*j0;  H.at<double>(1,1) += j1*j1;
        H.at<double>(1,2) += j1*j2;  H.at<double>(1,3) += j1*j3;

        H.at<double>(2,0) += j2*j0;  H.at<double>(2,1) += j2*j1;
        H.at<double>(2,2) += j2*j2;  H.at<double>(2,3) += j2*j3;

        H.at<double>(3,0) += j3*j0;  H.at<double>(3,1) += j3*j1;
        H.at<double>(3,2) += j3*j2;  H.at<double>(3,3) += j3*j3;
    }

    const int n = selected_jacobian_y.dim(0).extent();
    for (int i = 0; i < n; i++)
    {
        // J in [4], each is float
        double j0 = selected_jacobian_y(i, 0);
        double j1 = selected_jacobian_y(i, 1);
        double j2 = selected_jacobian_y(i, 2);
        double j3 = selected_jacobian_y(i, 3);

        // Outer product Jᵀ*J, accumulate into H
        // H is symmetrical, but we’ll just fill all entries for clarity:
        H.at<double>(0,0) += j0*j0;  H.at<double>(0,1) += j0*j1;
        H.at<double>(0,2) += j0*j2;  H.at<double>(0,3) += j0*j3;

        H.at<double>(1,0) += j1*j0;  H.at<double>(1,1) += j1*j1;
        H.at<double>(1,2) += j1*j2;  H.at<double>(1,3) += j1*j3;

        H.at<double>(2,0) += j2*j0;  H.at<double>(2,1) += j2*j1;
        H.at<double>(2,2) += j2*j2;  H.at<double>(2,3) += j2*j3;

        H.at<double>(3,0) += j3*j0;  H.at<double>(3,1) += j3*j1;
        H.at<double>(3,2) += j3*j2;  H.at<double>(3,3) += j3*j3;
    }

    return H;
}

bool VideoAligner::AlignNextFrame(
    const cv::Mat& inputFrame,
    SimilarityTransform& transform,
    bool phase_correlate)
{
    transform = SimilarityTransform(); // Identity transform

    if (!ComputePyramid(inputFrame)) {
        return false;
    }

    if (CurrFrameIndex == KeyframeIndex) {
        //uint64_t t0 = get_time_since_boot_microseconds();
        if (!ComputeKeyFrame()) {
            LastWidth = -1;
            return false;
        }
        //uint64_t t1 = get_time_since_boot_microseconds();
        //std::cout << "Keyframe setup took " << (t1 - t0) / 1000.f << " milliseconds\n";
    }

    if (phase_correlate) {
        //uint64_t t0 = get_time_since_boot_microseconds();

        cv::Point2d detected_shift;
        double response = 0.0;
        detected_shift = cv::phaseCorrelate(PhaseImage[PrevFrameIndex], PhaseImage[CurrFrameIndex], cv::noArray(), &response);
        if (response > 0.5) {
            const float phase_layer_scale = (1 << PhaseLevel) / float(1 << PyramidLevels);
            transform.TX = detected_shift.x * phase_layer_scale;
            transform.TY = detected_shift.y * phase_layer_scale;
            if (CurrFrameIndex == KeyframeIndex) {
                transform.TX = -transform.TX;
                transform.TY = -transform.TY;
            }
        }

        //uint64_t t1 = get_time_since_boot_microseconds();
        //std::cout << "Detected Shift: (" << detected_shift.x << ", " << detected_shift.y << ")  Response=" << response << " in " << (t1 - t0) << " microseconds" << std::endl;
    }

    for (int i = PyramidLevels - 1; i >= 0; i--) {
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

        //uint64_t t0 = get_time_since_boot_microseconds();

        if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_x, transform, warpdiff_x)) {
            std::cerr << "Failed to compute warp diff at level " << i << std::endl;
            return false;
        }
        if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_y, transform, warpdiff_y)) {
            std::cerr << "Failed to compute warp diff at level " << i << std::endl;
            return false;
        }

        // At the start of each pyramid level, find the argmax pixels that
        // are the closest between the template and the keyframe.
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

        // Find the Subset of the DeltaPixels with the smallest abs_delta
        const float smallest_fraction = 0.5f;
        const size_t selected_count_x =
            static_cast<size_t>(DeltaPixelsX.size() * smallest_fraction);
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
            static_cast<size_t>(DeltaPixelsY.size() * smallest_fraction);
        std::nth_element(
            DeltaPixelsY.begin(),
            DeltaPixelsY.begin() + selected_count_y,
            DeltaPixelsY.end(),
            [](const DeltaPixel &lhs, const DeltaPixel &rhs) {
                return lhs.abs_delta < rhs.abs_delta;
            }
        );
        DeltaPixelsY.resize(selected_count_y);

        if (selected_pixels_x.dimensions() != 2 ||
            selected_pixels_x.dim(0).extent() != selected_count_x ||
            selected_pixels_x.dim(1).extent() != 2)
        {
            selected_pixels_x = Halide::Runtime::Buffer<uint16_t>(selected_count_x, 2);
        }
        if (selected_pixels_y.dimensions() != 2 ||
            selected_pixels_y.dim(0).extent() != selected_count_y ||
            selected_pixels_y.dim(1).extent() != 2)
        {
            selected_pixels_y = Halide::Runtime::Buffer<uint16_t>(selected_count_y, 2);
        }
        if (selected_jacobian_x.dimensions() != 2 ||
            selected_jacobian_x.dim(0).extent() != selected_count_x ||
            selected_jacobian_x.dim(1).extent() != 4)
        {
            selected_jacobian_x = Halide::Runtime::Buffer<float>(selected_count_x, 4);
        }
        if (selected_jacobian_y.dimensions() != 2 ||
            selected_jacobian_y.dim(0).extent() != selected_count_y ||
            selected_jacobian_y.dim(1).extent() != 4)
        {
            selected_jacobian_y = Halide::Runtime::Buffer<float>(selected_count_y, 4);
        }

        for (size_t j = 0; j < selected_count_x; j++) {
            int tile_x = DeltaPixelsX[j].tile_x;
            int tile_y = DeltaPixelsX[j].tile_y;

            selected_pixels_x(j, 0) = grad_argmax_x(tile_x, tile_y, 0);
            selected_pixels_x(j, 1) = grad_argmax_x(tile_x, tile_y, 1);
            for (int k = 0; k < 4; k++) {
                selected_jacobian_x(j, k) = jacobian_x(tile_x, tile_y, k);
            }
        }
        for (size_t j = 0; j < selected_count_y; j++) {
            int tile_x = DeltaPixelsY[j].tile_x;
            int tile_y = DeltaPixelsY[j].tile_y;

            selected_pixels_y(j, 0) = grad_argmax_y(tile_x, tile_y, 0);
            selected_pixels_y(j, 1) = grad_argmax_y(tile_x, tile_y, 1);
            for (int k = 0; k < 4; k++) {
                selected_jacobian_y(j, k) = jacobian_y(tile_x, tile_y, k);
            }
        }

        cv::Mat H = ComputeHessianFromSelected(selected_jacobian_x, selected_jacobian_y);
        cv::Mat Hinv = H.inv(cv::DECOMP_SVD);

        //std::cout << "i=" << i << ": Selected pixels: " << selected_pixels.width() << std::endl;

        //uint64_t t1 = get_time_since_boot_microseconds();
        //std::cout << "Pyramid level " << i << " setup took " << (t1 - t0) / 1000.f << " milliseconds\n";

        const int max_iters = 64;
        for (int iter = 0; iter < max_iters; iter++) {
            //std::cout << "i=" << i << ": Iteration " << iter << "/" << max_iters << std::endl;
            //std::cout << "Transform: " << transform.toString() << std::endl;

            //uint64_t s0 = get_time_since_boot_microseconds();

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
                return false;
            }

            cv::Mat dt = Hinv * halide_vec4_to_mat(IcaResult);

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

            transform = delta_transform.compose(transform);

            //uint64_t s1 = get_time_since_boot_microseconds();
            //std::cout << "Pyramid level " << i << " ICA iteration took " << (s1 - s0) / 1000.f << " milliseconds\n";
 
            //std::cout << "dt = " << dt << std::endl;
            //std::cout << "New Transform: " << transform.toString() << std::endl;

            // FIXME: Add check for divergence

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

            /*
                There is a sweet spot for this threshold.
                Too low: Will iterate too many times, accumulating errors until it diverges.
                Too high: Will iterate too few times, creating visual errors and/or diverging more.
            */
            const double threshold = 0.03;
            if (displacement < threshold) {
                break;
            }

            if (iter >= max_iters - 1) {
                return false;
            }
        }

        if (i > 0) {
            // Move from half-resolution to full-resolution
            transform.TX *= 2.f;
            transform.TY *= 2.f;
        }
    }

    if (CurrFrameIndex != KeyframeIndex) {
        transform = transform.inverse();
    }

    return true;
}
