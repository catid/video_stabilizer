#include "ukf.hpp"

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

CameraMotionUKF::CameraMotionUKF()
{
    // Initialize UKF hyper-parameters
    alpha_ = 1e-3;
    beta_  = 2.0;
    kappa_ = 0.0;
    const int L = STATE_DIM;
    lambda_ = alpha_ * alpha_ * (L + kappa_) - L;

    // Initial state
    x_ = Eigen::VectorXd::Zero(STATE_DIM);
    // Initial covariance
    P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-2;

    // Example: set process noise Q_ for a constant-velocity model
    Q_ = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    // Heuristics: small noise on position, bigger on velocity, tune these as needed
    Q_(0,0) = 1e-3; // A
    Q_(1,1) = 1e-3; // B
    Q_(2,2) = 1e-2; // TX
    Q_(3,3) = 1e-2; // TY
    Q_(4,4) = 1e-3; // velocity A
    Q_(5,5) = 1e-3; // velocity B
    Q_(6,6) = 1e-2; // vx
    Q_(7,7) = 1e-2; // vy

    R_ = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    R_(0,0) = 1e-3;
    R_(1,1) = 1e-3;
    R_(2,2) = 1e-2;
    R_(3,3) = 1e-2;
}

//------------------------------------------------------------------------------
// UKF update
//------------------------------------------------------------------------------

SimilarityTransform CameraMotionUKF::update(const SimilarityTransform &meas, bool reset)
{
    // 3) If reset requested, reset the filter to the measurement
    if (reset) {
        x_(0) = meas.A;
        x_(1) = meas.B;
        x_(2) = meas.TX;
        x_(3) = meas.TY;
        x_(4) = 0.0;
        x_(5) = 0.0;
        x_(6) = 0.0;
        x_(7) = 0.0;
        P_ = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM); // 8×8
        P_(0,0) = 1e-3;
        P_(1,1) = 1e-3;
        P_(2,2) = 1e-2;
        P_(3,3) = 1e-2;
        P_(4,4) = 1e-3;
        P_(5,5) = 1e-3;
        P_(6,6) = 1e-2;
        P_(7,7) = 1e-2;
    }

    // UKF steps:
    //   a) Generate sigma points
    std::vector<Eigen::VectorXd> sigmaPoints;
    generateSigmaPoints(x_, P_, sigmaPoints);

    //   b) Predict each sigma point with the motion model
    //      we assume dt=1.0 for "frame-to-frame" (or you can pass your actual dt)
    double dt = 1.0; 
    std::vector<Eigen::VectorXd> sigmaPointsPred(sigmaPoints.size());
    for (size_t i=0; i<sigmaPoints.size(); i++) {
        sigmaPointsPred[i] = predictState(sigmaPoints[i], dt);
    }

    //   c) Compute predicted mean & covariance
    const int L = STATE_DIM;
    int nSigma = 2*L + 1;

    // Weights
    Eigen::VectorXd wMean(nSigma), wCov(nSigma);
    wMean(0) = lambda_ / (L + lambda_);
    wCov(0)  = lambda_ / (L + lambda_) + (1 - alpha_*alpha_ + beta_);
    for (int i=1; i<nSigma; i++) {
        wMean(i) = 1.0 / (2.0 * (L + lambda_));
        wCov(i)  = 1.0 / (2.0 * (L + lambda_));
    }

    // Predict state mean
    Eigen::VectorXd xPred = Eigen::VectorXd::Zero(L);
    for (int i=0; i<nSigma; i++) {
        xPred += wMean(i) * sigmaPointsPred[i];
    }

    // Predict state covariance
    Eigen::MatrixXd PPred = Eigen::MatrixXd::Zero(L, L);
    for (int i=0; i<nSigma; i++) {
        Eigen::VectorXd diff = sigmaPointsPred[i] - xPred;
        PPred += wCov(i) * (diff * diff.transpose());
    }
    // Add process noise
    PPred += Q_;

    //   d) Transform predicted sigma points into measurement space
    std::vector<Eigen::VectorXd> Zsigma(nSigma);
    for (int i=0; i<nSigma; i++) {
        const auto& state = sigmaPointsPred[i];
        Eigen::VectorXd z(MEAS_DIM);
        z(0) = state(0);
        z(1) = state(1);
        z(2) = state(2);
        z(3) = state(3);
        Zsigma[i] = z;
    }

    // Compute predicted measurement zPred
    Eigen::VectorXd zPred = Eigen::VectorXd::Zero(MEAS_DIM);
    for (int i=0; i<nSigma; i++) {
        zPred += wMean(i) * Zsigma[i];
    }

    // Compute measurement covariance S and cross-covariance Tc
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(L, MEAS_DIM);

    for (int i=0; i<nSigma; i++) {
        // measurement diff
        Eigen::VectorXd zDiff = Zsigma[i] - zPred;

        // state diff
        Eigen::VectorXd xDiff = sigmaPointsPred[i] - xPred;

        S  += wCov(i) * zDiff * zDiff.transpose();
        Tc += wCov(i) * xDiff * zDiff.transpose();
    }
    // Add measurement noise
    S += R_;

    //   e) Kalman gain
    Eigen::MatrixXd K = Tc * S.inverse();

    //   f) Update state mean/covariance with measurement
    Eigen::VectorXd zMeas(MEAS_DIM);
    zMeas << meas.A, meas.B, meas.TX, meas.TY;
    // measurement residual
    Eigen::VectorXd y = zMeas - zPred;

    // new state
    Eigen::VectorXd xNew = xPred + K * y;

    // new covariance
    Eigen::MatrixXd PNew = PPred - K * S * K.transpose();

    // store
    x_ = xNew;
    P_ = PNew;

    // 4) Build the final output transform from the filtered (r, tx, ty) plus the *original* scale
    SimilarityTransform currentFiltered;
    currentFiltered.A = x_(0);
    currentFiltered.B = x_(1);
    currentFiltered.TX = x_(2);
    currentFiltered.TY = x_(3);

    // 6) Return the *previous* result
    return currentFiltered;
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

void CameraMotionUKF::generateSigmaPoints(const Eigen::VectorXd &x,
                                          const Eigen::MatrixXd &P,
                                          std::vector<Eigen::VectorXd> &sigmaPoints)
{
    const int L = STATE_DIM;
    Eigen::MatrixXd A = P.llt().matrixL(); // chol decomp
    sigmaPoints.resize(2*L + 1, Eigen::VectorXd::Zero(L));

    // Calculate sigma points
    double scale = std::sqrt(L + lambda_);
    sigmaPoints[0] = x;
    for (int i=0; i<L; i++) {
        sigmaPoints[i+1]     = x + scale * A.col(i);
        sigmaPoints[i+1+L]   = x - scale * A.col(i);
    }
}

Eigen::VectorXd CameraMotionUKF::predictState(const Eigen::VectorXd &state, double dt)
{
    // State = [r, tx, ty, rVel, txVel, tyVel]
    Eigen::VectorXd xPred(STATE_DIM);
    double A     = state(0);
    double B     = state(1);
    double TX    = state(2);
    double TY    = state(3);
    double vA    = state(4);
    double vB    = state(5);
    double vTX   = state(6);
    double vTY   = state(7);

    A += vA * dt;
    B += vB * dt;
    TX += vTX * dt;
    TY += vTY * dt;

    xPred << A, B, TX, TY, vA, vB, vTX, vTY;
    return xPred;
}

Eigen::VectorXd CameraMotionUKF::stateToMeasurement(const Eigen::VectorXd &state)
{
    Eigen::VectorXd z(MEAS_DIM);
    z(0) = state(0);
    z(1) = state(1);
    z(2) = state(2);
    z(3) = state(3);
    return z;
}
