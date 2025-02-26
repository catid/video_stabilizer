#pragma once

#include "imgproc.hpp"

#include <Eigen/Dense>
#include <deque>
#include <cmath>

class CameraMotionUKF_FixedLag
{
public:
    static const int SINGLE_STATE_DIM = 8;  // [A,B,TX,TY, vA, vB, vTX, vTY]
    static const int MEAS_DIM = 4;         // measuring [A,B,TX,TY] only

    CameraMotionUKF_FixedLag(int lag);

    // update() takes a new measurement and returns the smoothed output
    SimilarityTransform update(const SimilarityTransform &meas, bool reset=false);

private:
    // Number of states in our augmented vector
    int lag_;                // e.g. N
    int augStateDim_;        // = lag_ * SINGLE_STATE_DIM

    // UKF hyperparameters
    double alpha_;
    double beta_;
    double kappa_;
    double lambda_;

    // Augmented state vector X_ and covariance P_
    // (dimension = lag_ * SINGLE_STATE_DIM)
    Eigen::VectorXd X_;
    Eigen::MatrixXd P_;

    // Process and measurement noise for augmented system
    Eigen::MatrixXd Q_aug_;
    Eigen::MatrixXd R_;

    // Weights for unscented transform
    Eigen::VectorXd wMean_;
    Eigen::VectorXd wCov_;

    // Some optional queue to hold "fully smoothed" outputs
    std::deque<SimilarityTransform> delayedOutput_;

private:
    // Helper methods
    void   generateSigmaPoints(const Eigen::VectorXd &x,
                               const Eigen::MatrixXd &P,
                               std::vector<Eigen::VectorXd> &sigmaPoints);
    Eigen::VectorXd predictAugState(const Eigen::VectorXd &X_aug);
    Eigen::VectorXd stateToMeasurement(const Eigen::VectorXd &X_aug);
    void ResetP();
};
