#pragma once

#include <Halide.h>
#include <opencv2/opencv.hpp>

#include "constants.h"

bool SparseJacobian(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y);

bool PyrDown(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<uint8_t>& output);

bool GradXY(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y);

// Returns floor(grad_x/tile_size) sized image with one (x,y) pair per tile,
// representing the maximum gradient magnitude position in pixels from image origin.
bool GradArgMax(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    int& tile_size,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y);

struct Point {
    double x = 0.0, y = 0.0;

    double distance(const Point& p) const;
};

struct SimilarityTransform {
    // Expr W_x = (1.0f + A) * x - B * y + TX;
    // Expr W_y = B * x + (1.0f + A) * y + TY;
    // So (A=0, B=0, TX=0, TY=0) is identity
    // Assumes upper left corner is (0,0), +x right, +y down
    // Units are in pixels
    double A = 0.0, B = 0.0, TX = 0.0, TY = 0.0;

    std::string toString() const;
    SimilarityTransform inverse() const;

    Point warp(Point p) const;
    double maxCornerDisplacement(double width, double height) const;

    // "this" = T1, param = T2
    // We want T3 = T2( T1(p) ) => apply T1, then T2.
    SimilarityTransform compose(
        const SimilarityTransform &w2) const;
    
};

inline SimilarityTransform difference(
    const SimilarityTransform &observed,
    const SimilarityTransform &expected)
{
    // 1. Invert the expected transform
    SimilarityTransform invExp = expected.inverse();

    // 2. "Subtract" = compose observed with the inverse of expected
    //    but recall the signature: T1.compose(T2) = T2 o T1
    //    so we want T_res = T_observed o T_expected^-1
    //    => that means T_res = invExp.compose(observed).
    return invExp.compose(observed);
}

bool ImageWarp(
    Halide::Runtime::Buffer<uint8_t>& input,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<float>& output);

Halide::Runtime::Buffer<uint8_t> mat_to_halide_buffer_u8(const cv::Mat &mat);
Halide::Runtime::Buffer<uint8_t> bgr_mat_to_halide_buffer_u8(const cv::Mat &mat);
cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<uint8_t> &buffer);
cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<float> &buffer);
cv::Mat halide_vec4_to_mat(const Halide::Runtime::Buffer<double> &vec4);

// Solve for parameter updates using sparse ICA:
// Sum( jac[i] * ( template[i] - Warped{keyframe[i]} ) )
bool SparseICA(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_x,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_y,
    Halide::Runtime::Buffer<float>& selected_jacobians_x,
    Halide::Runtime::Buffer<float>& selected_jacobians_y,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<double>& output);

bool SparseWarpDiff(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& local_max,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<uint16_t>& output);

cv::Mat warpBySimilarityTransform(const cv::Mat& src, const SimilarityTransform& transform);
