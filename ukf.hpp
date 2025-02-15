#pragma once

#include "imgproc.hpp"

#include <Eigen/Dense>
#include <deque>
#include <cmath>

class CameraMotionUKF {
public:
    CameraMotionUKF();
    
    /**
     * @brief Update the filter with the newest measured SimilarityTransform from frame i to frame i+1.
     * @param meas   The measured transform from the frame aligner (possibly noisy).
     * @param reset  If true, reset filter state to measurement. (e.g. if a jump or scene cut is detected)
     * @return       The *one-frame-delayed* smoothed transform from the previous step.
     */
    SimilarityTransform update(const SimilarityTransform &meas, bool reset);

private:
    static constexpr int STATE_DIM = 8; 
    static constexpr int MEAS_DIM  = 4;

    // UKF parameters
    double alpha_;
    double beta_;
    double kappa_;
    double lambda_;

    // State vector and covariance
    Eigen::VectorXd x_; // [A, B, TX, TY, vA, vB, vTX, vTY]
    Eigen::MatrixXd P_; // State covariance (8x8)

    // Process and measurement noise covariances
    Eigen::MatrixXd Q_; // (8x8)
    Eigen::MatrixXd R_; // (4x4)

private:
    // Helper: generate sigma points
    void generateSigmaPoints(const Eigen::VectorXd &x, 
                             const Eigen::MatrixXd &P, 
                             std::vector<Eigen::VectorXd> &sigmaPoints);

    // Helper: predict each sigma point through the motion model
    Eigen::VectorXd predictState(const Eigen::VectorXd &state, double dt);

    // Helper: compute measurement from predicted state (extract [r, tx, ty])
    Eigen::VectorXd stateToMeasurement(const Eigen::VectorXd &state);
};
