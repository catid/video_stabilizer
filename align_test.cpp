#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
using namespace std;

#include "imgproc.hpp"
#include "tools.hpp"
#include "alignment.hpp"


//------------------------------------------------------------------------
// Test PyrDown

void writeFloatMatToImage(const cv::Mat& floatMat, const std::string& outputFile) {
    // Check that the input matrix is of type CV_32F
    if (floatMat.type() != CV_32F) {
        throw std::invalid_argument("Input matrix must be of type CV_32F.");
    }

    // Normalize the matrix to the range [0, 255]
    cv::Mat normalized;
    cv::normalize(floatMat, normalized, 0, 255, cv::NORM_MINMAX);

    // Convert to CV_8U (8-bit unsigned integer)
    cv::Mat outputImage;
    normalized.convertTo(outputImage, CV_8U);

    // Write to disk
    if (!cv::imwrite(outputFile, outputImage)) {
        std::cerr << "Error: Could not write the image to disk." << std::endl;
    } else {
        std::cout << "Image saved to " << outputFile << std::endl;
    }
}

// Helper function to compute expected shift based on pyramid level
cv::Point2f computeExpectedShift(const SimilarityTransform& transform, int level) {
    return cv::Point2f(transform.TX, transform.TY);
}

void TestPyrDown()
{
    cv::Mat color_image = cv::imread("../input.png");
    if (color_image.empty()) {
        std::cerr << "Error: Could not load input image '../input.png'" << std::endl;
        return;
    }

    cv::Mat grayscale_image;
    cv::cvtColor(color_image, grayscale_image, cv::COLOR_BGR2GRAY);
    if (grayscale_image.empty()) {
        std::cerr << "Error: Grayscale conversion failed." << std::endl;
        return;
    }

    const int width = grayscale_image.cols;
    const int height = grayscale_image.rows;

    auto scale0 = mat_to_halide_buffer_u8(grayscale_image);
    auto scale1 = Halide::Runtime::Buffer<uint8_t>(width / 2, height / 2);
    auto scale2 = Halide::Runtime::Buffer<uint8_t>(width / 4, height / 4);
    auto scale3 = Halide::Runtime::Buffer<uint8_t>(width / 8, height / 8);
    auto scale4 = Halide::Runtime::Buffer<uint8_t>(width / 16, height / 16);
    auto scale5 = Halide::Runtime::Buffer<uint8_t>(width / 32, height / 32);

    // Gradient buffers
    auto scale0_gx = Halide::Runtime::Buffer<float>(scale0.width(), scale0.height());
    auto scale0_gy = Halide::Runtime::Buffer<float>(scale0.width(), scale0.height());
    auto scale1_gx = Halide::Runtime::Buffer<float>(scale1.width(), scale1.height());
    auto scale1_gy = Halide::Runtime::Buffer<float>(scale1.width(), scale1.height());
    auto scale2_gx = Halide::Runtime::Buffer<float>(scale2.width(), scale2.height());
    auto scale2_gy = Halide::Runtime::Buffer<float>(scale2.width(), scale2.height());
    auto scale3_gx = Halide::Runtime::Buffer<float>(scale3.width(), scale3.height());
    auto scale3_gy = Halide::Runtime::Buffer<float>(scale3.width(), scale3.height());
    auto scale4_gx = Halide::Runtime::Buffer<float>(scale4.width(), scale4.height());
    auto scale4_gy = Halide::Runtime::Buffer<float>(scale4.width(), scale4.height());
    auto scale5_gx = Halide::Runtime::Buffer<float>(scale5.width(), scale5.height());
    auto scale5_gy = Halide::Runtime::Buffer<float>(scale5.width(), scale5.height());

    // Create image pyramid
    PyrDown(scale0, scale1);
    PyrDown(scale1, scale2);
    PyrDown(scale2, scale3);
    PyrDown(scale3, scale4);
    PyrDown(scale4, scale5);

    // Convert Halide buffers back to cv::Mat for saving
    auto out_image0 = halide_buffer_to_mat(scale0);
    auto out_image1 = halide_buffer_to_mat(scale1);
    auto out_image2 = halide_buffer_to_mat(scale2);
    auto out_image3 = halide_buffer_to_mat(scale3);
    auto out_image4 = halide_buffer_to_mat(scale4);
    auto out_image5 = halide_buffer_to_mat(scale5);

    // Save the pyramid images
    cv::imwrite("out_image0.png", out_image0);
    cv::imwrite("out_image1.png", out_image1);
    cv::imwrite("out_image2.png", out_image2);
    cv::imwrite("out_image3.png", out_image3);
    cv::imwrite("out_image4.png", out_image4);
    cv::imwrite("out_image5.png", out_image5);

    // Compute gradients
    GradXY(scale0, scale0_gx, scale0_gy);
    GradXY(scale1, scale1_gx, scale1_gy);
    GradXY(scale2, scale2_gx, scale2_gy);
    GradXY(scale3, scale3_gx, scale3_gy);
    GradXY(scale4, scale4_gx, scale4_gy);
    GradXY(scale5, scale5_gx, scale5_gy);

    // Save gradient images
    writeFloatMatToImage(halide_buffer_to_mat(scale0_gx), "scale0_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale0_gy), "scale0_gy.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale1_gx), "scale1_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale1_gy), "scale1_gy.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale2_gx), "scale2_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale2_gy), "scale2_gy.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale3_gx), "scale3_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale3_gy), "scale3_gy.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale4_gx), "scale4_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale4_gy), "scale4_gy.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale5_gx), "scale5_gx.png");
    writeFloatMatToImage(halide_buffer_to_mat(scale5_gy), "scale5_gy.png");

    // Prepare warp buffers
    auto warp0 = Halide::Runtime::Buffer<float>(width, height);
    auto warp1 = Halide::Runtime::Buffer<float>(width / 2, height / 2);
    auto warp2 = Halide::Runtime::Buffer<float>(width / 4, height / 4);
    auto warp3 = Halide::Runtime::Buffer<float>(width / 8, height / 8);
    auto warp4 = Halide::Runtime::Buffer<float>(width / 16, height / 16);
    auto warp5 = Halide::Runtime::Buffer<float>(width / 32, height / 32);

    // Define the similarity transform
    SimilarityTransform transform{0.f, 0.f, 4.f, 4.f};

    float max_displacement = transform.maxCornerDisplacement(width, height);

    double Cx = color_image.cols / 2.0; // computed earlier; color_image still in scope
    double Cy = color_image.rows / 2.0;
    auto transform_inv = transform.inverse(Cx, Cy);

    float max_displacement_inv = transform.maxCornerDisplacement(width, height);

    std::cout << "Max displacement: " << max_displacement << std::endl;
    std::cout << "Max displacement (inverse): " << max_displacement_inv << std::endl;

    // Apply warping to each scale
    ImageWarp(scale0, transform_inv, scale0.width(), scale0.height(), warp0);
    ImageWarp(scale1, transform_inv, scale1.width(), scale1.height(), warp1);
    ImageWarp(scale2, transform_inv, scale2.width(), scale2.height(), warp2);
    ImageWarp(scale3, transform_inv, scale3.width(), scale3.height(), warp3);
    ImageWarp(scale4, transform_inv, scale4.width(), scale4.height(), warp4);
    ImageWarp(scale5, transform_inv, scale5.width(), scale5.height(), warp5);

    // Save warped images
    writeFloatMatToImage(halide_buffer_to_mat(warp0), "warp0.png");
    writeFloatMatToImage(halide_buffer_to_mat(warp1), "warp1.png");
    writeFloatMatToImage(halide_buffer_to_mat(warp2), "warp2.png");
    writeFloatMatToImage(halide_buffer_to_mat(warp3), "warp3.png");
    writeFloatMatToImage(halide_buffer_to_mat(warp4), "warp4.png");
    writeFloatMatToImage(halide_buffer_to_mat(warp5), "warp5.png");

    // Verification using phaseCorrelate
    // Store original and warped images in a vector for easy iteration
    std::vector<std::pair<Halide::Runtime::Buffer<uint8_t>, Halide::Runtime::Buffer<float>>> image_pairs = {
        {scale0, warp0},
        {scale1, warp1},
        {scale2, warp2},
        {scale3, warp3},
        {scale4, warp4},
        {scale5, warp5}
    };

    for (size_t level = 0; level < image_pairs.size(); ++level) {
        // Convert Halide buffers to cv::Mat
        cv::Mat original = halide_buffer_to_mat(image_pairs[level].first);
        cv::Mat warped = halide_buffer_to_mat(image_pairs[level].second);

        // Ensure both images are of type CV_32F for phaseCorrelate
        cv::Mat original_float, warped_float;
        original.convertTo(original_float, CV_32F);
        warped.convertTo(warped_float, CV_32F);

        // Apply windowing if necessary (e.g., Hanning window)
        // For simplicity, we'll skip windowing here

        // Compute the shift using phaseCorrelate
        cv::Point2d detected_shift;
        double response = 0.0;
        detected_shift = cv::phaseCorrelate(original_float, warped_float, cv::noArray(), &response);

        // Compute expected shift for this level
        cv::Point2f expected_shift = computeExpectedShift(transform, level);

        // Log the results
        std::cout << "Level " << level << ":\n";
        std::cout << "  Expected Shift: (" << expected_shift.x << ", " << expected_shift.y << ")\n";
        std::cout << "  Detected Shift: (" << detected_shift.x << ", " << detected_shift.y << ")\n";
        std::cout << "  Response: " << response << "\n";

        // Optionally, verify if the detected shift is within an acceptable range
        float tolerance = 0.5f; // Define a suitable tolerance
        if (std::abs(detected_shift.x - expected_shift.x) > tolerance ||
            std::abs(detected_shift.y - expected_shift.y) > tolerance) {
            std::cerr << "Warning: Detected shift at level " << level << " differs from expected shift.\n";
        } else {
            std::cout << "  Shift verification passed.\n";
        }
    }

    uint64_t t0 = get_time_since_boot_microseconds();

    int tile0_size = -1;
    Halide::Runtime::Buffer<uint16_t> local_max0_x, local_max0_y;
    if (!GradArgMax(scale0_gx, scale0_gy, tile0_size, local_max0_x, local_max0_y)) {
        std::cerr << "Failed to compute argmax" << std::endl;
        return;
    }

    int tile1_size = -1;
    Halide::Runtime::Buffer<uint16_t> local_max1_x, local_max1_y;
    if (!GradArgMax(scale1_gx, scale1_gy, tile1_size, local_max1_x, local_max1_y)) {
        std::cerr << "Failed to compute argmax" << std::endl;
        return;
    }

    uint64_t t1 = get_time_since_boot_microseconds();
    std::cout << "Argmax computation time: " << (t1 - t0) / 1000.0 << " ms" << std::endl;

    std::cout << "Scale 0: Argmax tile size: " << tile0_size << std::endl;
    std::cout << "Argmax dimensions: " << local_max0_x.dimensions() << std::endl;
    std::cout << "Argmax dimensions[0]: " << local_max0_x.dim(0).extent() << std::endl;
    std::cout << "Argmax dimensions[1]: " << local_max0_x.dim(1).extent() << std::endl;
    std::cout << "Argmax dimensions[2]: " << local_max0_x.dim(2).extent() << std::endl;
    std::cout << "Selected pixel count: " << local_max0_x.dim(0).extent() * local_max0_x.dim(1).extent() << std::endl;
    std::cout << "Tile(0,0): " << local_max0_x(0, 0, 0) << ", " << local_max0_x(0, 0, 1) << std::endl;
    std::cout << "Tile(1,1): " << local_max0_x(1, 1, 0) << ", " << local_max0_x(1, 1, 1) << std::endl;

    std::cout << "Scale 1: Argmax tile size: " << tile1_size << std::endl;
    std::cout << "Argmax dimensions: " << local_max1_x.dimensions() << std::endl;
    std::cout << "Argmax dimensions[0]: " << local_max1_x.dim(0).extent() << std::endl;
    std::cout << "Argmax dimensions[1]: " << local_max1_x.dim(1).extent() << std::endl;
    std::cout << "Argmax dimensions[2]: " << local_max1_x.dim(2).extent() << std::endl;
    std::cout << "Selected pixel count: " << local_max1_x.dim(0).extent() * local_max1_x.dim(1).extent() << std::endl;
    std::cout << "Tile(0,0): " << local_max1_x(0, 0, 0) << ", " << local_max1_x(0, 0, 1) << std::endl;
    std::cout << "Tile(1,1): " << local_max1_x(1, 1, 0) << ", " << local_max1_x(1, 1, 1) << std::endl;
}

static float EPSILON = 1e-5f; // Tolerance for floating comparisons

/**
 * \brief Compare two floating values with a small tolerance.
 */
inline bool nearlyEqual(float a, float b, float epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

/**
 * \brief Test that SimilarityTransform::inverse() really is the inverse.
 */
void TestSimilarityTransformInverse()
{
    // A few sample transforms to test
    // A, B, TX, TY
    std::vector<SimilarityTransform> transforms = {
        {0.f, 0.f, 0.f, 0.f},    // identity
        {0.1f, 0.f, 10.f, 20.f}, // scale+translate
        {0.f, 0.1f, 5.f, -5.f},  // rotate+translate
        {0.05f, 0.05f, 100.f, 50.f} // scale+rotate+translate
    };

    // We'll pick a few test points
    std::vector<Point> test_points = {
        {0.f, 0.f},
        {100.f, 100.f},
        {50.f, 200.f},
        {-10.f, 30.f},
        {1.3f, -2.7f}
    };

    for (size_t i = 0; i < transforms.size(); ++i) {
        const SimilarityTransform& T = transforms[i];
        SimilarityTransform Tinv = T.inverse();
        
        // Check that Tinv(T(p)) == p, for each p
        for (auto &p : test_points) {
            Point warped = T.warp(p);
            Point unwarped = Tinv.warp(warped);

            bool pass = nearlyEqual(p.x, unwarped.x) && nearlyEqual(p.y, unwarped.y);
            if (!pass) {
                std::cerr << "[FAIL] Inverse test: T=" << T.toString()
                          << ", p=(" << p.x << ", " << p.y << ")"
                          << ", T->p=(" << warped.x << ", " << warped.y << ")"
                          << ", Tinv->warped=(" << unwarped.x << ", " << unwarped.y << ")\n";
            }
            else {
                std::cout << "[PASS] Inverse test: T=" << T.toString()
                          << " on p=(" << p.x << "," << p.y << ")\n";
            }
        }
    }
}

/**
 * \brief Test that SimilarityTransform::compose() correctly composes transforms
 *
 * By definition, if T3 = T1.compose(T2),
 * then T3(p) should be T2( T1(p) ) for any point p.
 */
void TestSimilarityTransformCompose()
{
    // Some example transforms
    SimilarityTransform T1 {0.1f,  0.0f,  10.f, 20.f}; // scale+translate
    SimilarityTransform T2 {0.f,   0.1f,   5.f,  5.f}; // rotate+translate
    SimilarityTransform T3 = T1.compose(T2); 

    std::cout << "T1: " << T1.toString() << "\n";
    std::cout << "T2: " << T2.toString() << "\n";
    std::cout << "T3 = T1.compose(T2): " << T3.toString() << "\n";

    // Test points
    std::vector<Point> test_points = {
        {0.f, 0.f},
        {10.f, 20.f},
        {50.f, 50.f},
        {-10.f, 30.f}
    };

    for (auto &p : test_points) {
        // Evaluate T3(p)
        Point p3 = T3.warp(p);
        // Evaluate T2( T1(p) )
        Point p1 = T1.warp(p);
        Point p2 = T2.warp(p1);

        bool pass = (nearlyEqual(p3.x, p2.x) && nearlyEqual(p3.y, p2.y));
        if (!pass) {
            std::cerr << "[FAIL] Compose test for p=(" << p.x << ", " << p.y << ")\n"
                      << "  T3(p)    = (" << p3.x << ", " << p3.y << ")\n"
                      << "  T2(T1(p))= (" << p2.x << ", " << p2.y << ")\n";
        } else {
            std::cout << "[PASS] Compose test for p=(" << p.x << ", " << p.y << ")\n";
        }
    }
}

/**
 * \brief Test that ImageWarp(...) matches the transform by checking a simple pattern.
 *
 * Creates a small synthetic image, warps it, and tries to detect the shift/rotation
 * using something like phase correlatation or known corner check.
 *
 * For thoroughness, this example does a shift test using OpenCV’s phaseCorrelate
 * (assuming it is purely a translation). If you want to test rotation as well, you
 * can do corner checks or more advanced matching.
 */
void TestImageWarpCorrectness()
{
    // 1) Create a synthetic single-channel 64x64 image with a bright square in the center
    const int W = 64, H = 64;
    cv::Mat synthetic(H, W, CV_8UC1, cv::Scalar(0));
    cv::rectangle(synthetic, cv::Rect(20, 20, 10, 10), cv::Scalar(255), cv::FILLED);

    // 2) Convert to Halide buffer
    auto synthetic_in = mat_to_halide_buffer_u8(synthetic);

    // 3) Pick a transform (pure translation here) for a basic test
    //    If you include rotation (B != 0), simple phaseCorrelate won't detect it as well,
    //    so you may want to do a different validation. For demonstration, let's do a shift.
    SimilarityTransform T {0.f, 0.f, 5.f, 7.f}; // shift x=5, y=7

    // 4) Warp the image
    Halide::Runtime::Buffer<float> warped_out(W, H);
    {
        auto Cx_s = synthetic_in.width() / 2;
        auto Cy_s = synthetic_in.height() / 2;
        ImageWarp(synthetic_in, T.inverse(Cx_s, Cy_s), synthetic_in.width(), synthetic_in.height(), warped_out);
    }
    // (We use T.inverse() if we interpret "output -> input" mapping. Adjust to your usage.)

    // 5) Convert back to CV_32F for phase correlation
    cv::Mat warped_mat = halide_buffer_to_mat(warped_out);
    cv::Mat in_float, warped_float;
    synthetic.convertTo(in_float, CV_32F);
    warped_mat.convertTo(warped_float, CV_32F);

    // 6) Use phaseCorrelate to detect shift
    double response = 0.0;
    cv::Point2d shift = cv::phaseCorrelate(in_float, warped_float, cv::noArray(), &response);

    std::cout << "TestImageWarpCorrectness:\n";
    std::cout << "  Actual transform shift: (" << T.TX << ", " << T.TY << ")\n";
    std::cout << "  Detected shift via phaseCorrelate: (" 
              << shift.x << ", " << shift.y << "), response=" << response << "\n";

    // 7) Check if the detected shift is close to the actual shift
    float tolerance = 0.5f;
    if (std::fabs(shift.x - T.TX) > tolerance || std::fabs(shift.y - T.TY) > tolerance) {
        std::cerr << "[FAIL] Warp shift does not match expected transform.\n";
    } else {
        std::cout << "[PASS] Warp shift matched expected transform (within tolerance).\n";
    }
}

/**
 * \brief Generate a random SimilarityTransform, with user-specified range on the parameters.
 *        For example, scale_range ~ [ -0.5 .. +0.5 ] => scale factor in [ 0.5 .. 1.5 ]
 *        B_range ~ [ -0.2 .. +0.2 ] => small rotations, etc.
 *        trans_range ~ [ -50 .. +50 ] => translations
 */
SimilarityTransform randomTransform(
        std::mt19937 &rng,
        double scaleMin, double scaleMax,
        double bMin, double bMax,
        double txMin, double txMax,
        double tyMin, double tyMax)
{
    std::uniform_real_distribution<double> scaleDist(scaleMin, scaleMax);
    std::uniform_real_distribution<double> bDist(bMin, bMax);
    std::uniform_real_distribution<double> tDistX(txMin, txMax);
    std::uniform_real_distribution<double> tDistY(tyMin, tyMax);

    SimilarityTransform T;
    // A => around scale-1. E.g. if A=0.2 => scale=1.2
    T.A  = scaleDist(rng);
    T.B  = bDist(rng);
    T.TX = tDistX(rng);
    T.TY = tDistY(rng);

    return T;
}

/**
 * \brief Generate a random 2D point within given bounding box
 */
Point randomPoint(std::mt19937 &rng, double xMin, double xMax, double yMin, double yMax)
{
    std::uniform_real_distribution<double> xDist(xMin, xMax);
    std::uniform_real_distribution<double> yDist(yMin, yMax);
    return { xDist(rng), yDist(rng) };
}


/**
 * \brief Test that T.inverse() is truly the inverse, for random T.
 */
void TestRandomizedInverse()
{
    std::mt19937 rng(12345); // fixed seed for reproducibility

    const int NUM_TESTS = 50;
    std::cout << "[RandomizedInverseTest] Testing " << NUM_TESTS << " random transforms...\n";

    for(int i=0; i < NUM_TESTS; ++i) {
        // Generate a random transform with "reasonable" ranges
        SimilarityTransform T = randomTransform(rng,
                                                -0.3, 0.3,   // A in [-0.3..0.3] => scale in [0.7..1.3]
                                                -0.2, 0.2,   // B in [-0.2..0.2]
                                                -50,  50,    // TX
                                                -50,  50);   // TY

        // Compute Tinv
        SimilarityTransform Tinv = T.inverse();

        // For each T, let's test a handful of random points
        for(int j=0; j < 10; ++j) {
            Point p = randomPoint(rng, -100, 100, -100, 100);
            Point warped = T.warp(p);
            Point unwarped = Tinv.warp(warped);

            // Check that unwarped ~ p
            if(! (nearlyEqual(unwarped.x, p.x) && nearlyEqual(unwarped.y, p.y))) {
                std::cerr << "FAIL: Inverse mismatch.\n"
                          << "  T  = " << T.toString() << "\n"
                          << "  p  = (" << p.x << ", " << p.y << ")\n"
                          << "  T->p = (" << warped.x << ", " << warped.y << ")\n"
                          << "  Tinv->(T->p) = (" << unwarped.x << ", " << unwarped.y << ")\n";
            }
        }
    }

    std::cout << "  [PASS] Completed Inverse checks.\n";
}


/**
 * \brief Test that T1.compose(T2) matches T2(T1(p)). Also test associativity:
 *        (T1 ∘ T2) ∘ T3 == T1 ∘ (T2 ∘ T3).
 */
void TestRandomizedCompose()
{
    std::mt19937 rng(6789);

    const int NUM_TESTS = 50;
    std::cout << "[RandomizedComposeTest] Testing " << NUM_TESTS << " random transforms...\n";

    for(int i=0; i<NUM_TESTS; ++i) {
        // Random T1, T2
        SimilarityTransform T1 = randomTransform(rng, -0.3,0.3, -0.2,0.2, -50,50, -50,50);
        SimilarityTransform T2 = randomTransform(rng, -0.3,0.3, -0.2,0.2, -50,50, -50,50);

        // Compose T3 = T1.compose(T2). By definition, T3(p) = T2(T1(p)) if your code's convention is that:
        //   T1.compose(T2) => apply T1 first, then T2
        SimilarityTransform T3 = T1.compose(T2);

        // Check T3(p) vs T2(T1(p)) for random points
        for(int j=0; j<5; ++j) {
            Point p = randomPoint(rng, -100,100, -100,100);
            Point p1 = T1.warp(p);
            Point p2 = T2.warp(p1);
            Point p3 = T3.warp(p);

            if(! (nearlyEqual(p2.x, p3.x) && nearlyEqual(p2.y, p3.y))) {
                std::cerr << "FAIL: Compose mismatch.\n"
                          << "  T1 = " << T1.toString() << "\n"
                          << "  T2 = " << T2.toString() << "\n"
                          << "  p  = (" << p.x << ", " << p.y << ")\n"
                          << "  T1->p   = (" << p1.x << ", " << p1.y << ")\n"
                          << "  T2->(T1->p) = (" << p2.x << ", " << p2.y << ")\n"
                          << "  T3->p        = (" << p3.x << ", " << p3.y << ")\n";
            }
        }
    }

    // Now let's do a small test for associativity: 
    //   (T1.compose(T2)).compose(T3) == T1.compose(T2.compose(T3))
    // For a group of transformations, we do have associativity in theory. 
    for(int i=0; i<NUM_TESTS; ++i) {
        SimilarityTransform T1 = randomTransform(rng, -0.3,0.3, -0.2,0.2, -50,50, -50,50);
        SimilarityTransform T2 = randomTransform(rng, -0.3,0.3, -0.2,0.2, -50,50, -50,50);
        SimilarityTransform T3 = randomTransform(rng, -0.3,0.3, -0.2,0.2, -50,50, -50,50);

        SimilarityTransform left  = (T1.compose(T2)).compose(T3);
        SimilarityTransform right =  T1.compose(T2.compose(T3));

        // Check random points
        for(int j=0; j<5; ++j) {
            Point p = randomPoint(rng, -100,100, -100,100);
            Point Lp = left.warp(p);
            Point Rp = right.warp(p);
            if(! (nearlyEqual(Lp.x, Rp.x) && nearlyEqual(Lp.y, Rp.y))) {
                std::cerr << "FAIL: Compose associativity mismatch.\n"
                          << "  T1 = " << T1.toString() << "\n"
                          << "  T2 = " << T2.toString() << "\n"
                          << "  T3 = " << T3.toString() << "\n"
                          << "  p  = (" << p.x << ", " << p.y << ")\n"
                          << "  (T1∘T2)∘T3->p = (" << Lp.x << ", " << Lp.y << ")\n"
                          << "  T1∘(T2∘T3)->p = (" << Rp.x << ", " << Rp.y << ")\n";
            }
        }
    }

    std::cout << "  [PASS] Completed Compose checks.\n";
}

/**
 * \brief Test the round-trip T.compose(T.inverse()) ~ identity
 *        and T.inverse().compose(T) ~ identity
 */
void TestInverseComposeIdentity()
{
    std::mt19937 rng(9999);

    const int NUM_TESTS = 50;
    std::cout << "[InverseComposeIdentity] Testing " << NUM_TESTS << " random transforms...\n";

    for(int i=0; i < NUM_TESTS; ++i) {
        SimilarityTransform T = randomTransform(rng,
                                                -0.3, 0.3,
                                                -0.2, 0.2,
                                                -50,  50,
                                                -50,  50);
        SimilarityTransform Ti = T.inverse();

        // Compose T∘Ti
        SimilarityTransform TTi = T.compose(Ti); // hopefully identity
        SimilarityTransform TiT = Ti.compose(T); // hopefully identity

        // Check random points
        for(int j=0; j<5; ++j) {
            Point p = randomPoint(rng, -100,100, -100,100);

            Point TTi_p = TTi.warp(p);
            Point TiT_p = TiT.warp(p);

            // Expect both ~ p
            if(! (nearlyEqual(TTi_p.x, p.x) && nearlyEqual(TTi_p.y, p.y))) {
                std::cerr << "FAIL: T∘Tinv != identity.\n"
                          << "  T = " << T.toString() << "\n"
                          << "  p = (" << p.x << ", " << p.y << ")\n"
                          << "  T∘Tinv->p = (" << TTi_p.x << ", " << TTi_p.y << ")\n";
            }

            if(! (nearlyEqual(TiT_p.x, p.x) && nearlyEqual(TiT_p.y, p.y))) {
                std::cerr << "FAIL: Tinv∘T != identity.\n"
                          << "  T = " << T.toString() << "\n"
                          << "  p = (" << p.x << ", " << p.y << ")\n"
                          << "  Tinv∘T->p = (" << TiT_p.x << ", " << TiT_p.y << ")\n";
            }
        }
    }

    std::cout << "  [PASS] Completed Round-Trip TcomposeTinverse identity checks.\n";
}

/**
 * \brief Runs all our similarity transform tests
 */
void TestSimilarityTransformsAll()
{
    std::cout << "=== Testing SimilarityTransform::inverse() ===\n";
    TestSimilarityTransformInverse();

    std::cout << "\n=== Testing SimilarityTransform::compose() ===\n";
    TestSimilarityTransformCompose();

    std::cout << "\n=== Testing ImageWarp correctness ===\n";
    TestImageWarpCorrectness();

    TestRandomizedInverse();
    TestRandomizedCompose();
    TestInverseComposeIdentity();
}

//------------------------------------------------------------------------
// AlignImagePair

int AlignImagePair()
{
    int m_frameIndex = 0;

    // 1) Load template (reference) image and input image.
    //    We assume template.png and input.png exist in the current directory.
    cv::Mat templateImg = cv::imread("../template.png", cv::IMREAD_COLOR);
    if (templateImg.empty()) {
        std::cerr << "Error: Could not load template.png\n";
        return 1;
    }

    cv::Mat inputImg = cv::imread("../input.png", cv::IMREAD_COLOR);
    if (inputImg.empty()) {
        std::cerr << "Error: Could not load input.png\n";
        return 1;
    }

    // 2) Create the VideoAligner instance
    VideoAligner aligner;

    // 3) First call: pass the template image.
    //    Internally, this should set it as the keyframe (depending on your code logic).
    SimilarityTransform transform1;
    bool success1 = aligner.AlignNextFrame(templateImg, transform1);
    m_frameIndex++;

    // 4) Second call: pass the input image to be aligned to the existing keyframe.
    SimilarityTransform transform2;
    bool success2 = aligner.AlignNextFrame(inputImg, transform2);
    m_frameIndex++;
    if (!success2) {
        std::cerr << "Frame " << m_frameIndex << ": Alignment failed.\n";
        return 1;
    } else {
        std::cout << "Frame " << m_frameIndex
                  << ": Alignment successful. Transform = "
                  << transform2.toString() << "\n";
    }

    // 5) Warp the input image according to the transform we just got
    //    (Assuming we have a function "ImageWarp(...)" that warps a single-channel or 3-channel image.)
    //    If your code only supports single-channel, convert first:
    cv::Mat inputGray;
    cv::cvtColor(inputImg, inputGray, cv::COLOR_BGR2GRAY);
    Halide::Runtime::Buffer<uint8_t> inputHalide = mat_to_halide_buffer_u8(inputGray);

    // We'll warp into a float buffer (as in your code). Adjust size if needed.
    Halide::Runtime::Buffer<float> warpedBuf(inputGray.cols, inputGray.rows);
    if (!ImageWarp(inputHalide, transform2, warpedBuf)) {
        std::cerr << "Error: ImageWarp failed.\n";
        return 1;
    }

    // Convert float buffer back to 8-bit for saving
    cv::Mat warpedMat = halide_buffer_to_mat(warpedBuf);  // => CV_32FC1
    warpedMat.convertTo(warpedMat, CV_8UC1);              // clamp/convert to grayscale 8-bit

    // 6) Write the warped output
    if (!cv::imwrite("../../aligned.png", warpedMat)) {
        std::cerr << "Error: Could not write aligned.png.\n";
        return 1;
    }
    std::cout << "Wrote aligned.png successfully.\n";

    return 0;
}

//------------------------------------------------------------------------
// Entrypoint

int main()
{
    TestPyrDown();
    TestSimilarityTransformsAll();
    AlignImagePair();
    return 0;
}
