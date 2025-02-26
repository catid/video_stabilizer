#include "ukf.hpp"

CameraMotionUKF_FixedLag::CameraMotionUKF_FixedLag(int lag)
  : lag_(lag)
{
    // The new dimension is lag_ * SINGLE_STATE_DIM
    augStateDim_ = lag_ * SINGLE_STATE_DIM;

    // Typical UKF hyperparams
    alpha_ = 1.0;
    beta_  = 2.0;
    kappa_ = 3.0 - (double)augStateDim_;
    lambda_ = alpha_*alpha_ * (augStateDim_ + kappa_) - augStateDim_;

    // Build weight vectors
    const int nSigma = 2 * augStateDim_ + 1;
    wMean_.resize(nSigma);
    wCov_.resize(nSigma);
    wMean_(0) = lambda_ / (augStateDim_ + lambda_);
    wCov_(0)  = wMean_(0) + (1.0 - alpha_*alpha_ + beta_);
    for (int i = 1; i < nSigma; i++) {
        wMean_(i) = 1.0 / (2.0 * (augStateDim_ + lambda_));
        wCov_(i)  = wMean_(i);
    }

    // Initialize the augmented state with zeros
    X_ = Eigen::VectorXd::Zero(augStateDim_);

    // Initialize the covariance to something small but nonzero
    ResetP();

    // Build Q_aug_ as block-diagonal or something that accounts
    // for each block's process noise. For simplicity, below we
    // assume each block has the same Q as your normal single-state Q,
    // then place it along the diagonal. You can tune as needed.
    Eigen::MatrixXd Q_single = Eigen::MatrixXd::Zero(SINGLE_STATE_DIM, SINGLE_STATE_DIM);
    Q_single(0,0) = 1e-2;
    Q_single(1,1) = 1e-2;
    Q_single(2,2) = 1e-1;
    Q_single(3,3) = 1e-1;
    Q_single(4,4) = 1e-2;
    Q_single(5,5) = 1e-2;
    Q_single(6,6) = 1e-1;
    Q_single(7,7) = 1e-1;

    Q_aug_ = Eigen::MatrixXd::Zero(augStateDim_, augStateDim_);
    for (int block = 0; block < lag_; block++) {
        int start = block * SINGLE_STATE_DIM;
        Q_aug_.block(start, start, SINGLE_STATE_DIM, SINGLE_STATE_DIM) = Q_single;
    }

    // Measurement noise R_ stays the same dimension
    // (measuring [A,B,TX,TY] for the *top block* only)
    R_ = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    R_(0,0) = 1e-3;
    R_(1,1) = 1e-3;
    R_(2,2) = 1e-2;
    R_(3,3) = 1e-2;
}

void CameraMotionUKF_FixedLag::ResetP()
{
    P_ = Eigen::MatrixXd::Zero(augStateDim_, augStateDim_);
    for (int i = 0; i < augStateDim_; i += SINGLE_STATE_DIM) {
        P_(i,i) = 1e-3;
        P_(i+1,i+1) = 1e-3;
        P_(i+2,i+2) = 1e-2;
        P_(i+3,i+3) = 1e-2;
        P_(i+4,i+4) = 1e-3;
        P_(i+5,i+5) = 1e-3;
        P_(i+6,i+6) = 1e-2;
        P_(i+7,i+7) = 1e-2;
    }
}

SimilarityTransform CameraMotionUKF_FixedLag::update(const SimilarityTransform &meas, bool reset)
{
    // 1) Optionally output the "fully smoothed" state that has just
    //    left our lag window. For example, if you want the oldest
    //    block in X_ (which is block index = lag_-1).
    SimilarityTransform out;
    if (!delayedOutput_.empty()) {
        out = delayedOutput_.front();
        delayedOutput_.pop_front();
    } else {
        out = SimilarityTransform(); // identity
    }

    // 2) If reset, then fill the entire augmented state with the current measurement
    //    and zero the velocities. This basically sets X_(0..7) repeated across all blocks.
    if (reset) {
        for (int block = 0; block < lag_; block++) {
            int idx = block * SINGLE_STATE_DIM;
            X_(idx+0) = meas.A;
            X_(idx+1) = meas.B;
            X_(idx+2) = meas.TX;
            X_(idx+3) = meas.TY;
            X_(idx+4) = 0.0;
            X_(idx+5) = 0.0;
            X_(idx+6) = 0.0;
            X_(idx+7) = 0.0;
        }
        ResetP();
    }

    // 3) Generate sigma points for augmented state
    std::vector<Eigen::VectorXd> sigmaPoints;
    generateSigmaPoints(X_, P_, sigmaPoints);
    const int nSigma = (int)sigmaPoints.size(); // = 2*augStateDim_ + 1

    // 4) Predict step: apply predictAugState() to each sigma point
    std::vector<Eigen::VectorXd> sigmaPointsPred(nSigma);
    for (int i = 0; i < nSigma; i++) {
        sigmaPointsPred[i] = predictAugState(sigmaPoints[i]);
    }

    // 5) Compute predicted mean XPred and covariance PPred
    Eigen::VectorXd XPred = Eigen::VectorXd::Zero(augStateDim_);
    for (int i = 0; i < nSigma; i++) {
        XPred += wMean_(i) * sigmaPointsPred[i];
    }

    Eigen::MatrixXd PPred = Eigen::MatrixXd::Zero(augStateDim_, augStateDim_);
    for (int i = 0; i < nSigma; i++) {
        Eigen::VectorXd diff = sigmaPointsPred[i] - XPred;
        PPred += wCov_(i) * diff * diff.transpose();
    }
    PPred += Q_aug_;

    // 6) Transform predicted sigma points into measurement space
    //    We only measure the top block => dimension = 4
    std::vector<Eigen::VectorXd> Zsigma(nSigma);
    for (int i = 0; i < nSigma; i++) {
        Zsigma[i] = stateToMeasurement(sigmaPointsPred[i]); // 4D
    }
    // Predicted measurement zPred
    Eigen::VectorXd zPred = Eigen::VectorXd::Zero(MEAS_DIM);
    for (int i = 0; i < nSigma; i++) {
        zPred += wMean_(i) * Zsigma[i];
    }

    // 7) Compute measurement covariance S and cross-covariance Tc
    Eigen::MatrixXd S  = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(augStateDim_, MEAS_DIM);
    for (int i = 0; i < nSigma; i++) {
        Eigen::VectorXd zDiff = Zsigma[i] - zPred;
        Eigen::VectorXd xDiff = sigmaPointsPred[i] - XPred;
        S  += wCov_(i) * zDiff * zDiff.transpose();
        Tc += wCov_(i) * xDiff * zDiff.transpose();
    }
    S += R_;

    // 8) Kalman gain
    Eigen::MatrixXd K = Tc * S.inverse();

    // 9) Update state
    Eigen::VectorXd zMeas(MEAS_DIM);
    zMeas << meas.A, meas.B, meas.TX, meas.TY;

    Eigen::VectorXd y = zMeas - zPred;
    Eigen::VectorXd XNew = XPred + K * y;
    Eigen::MatrixXd PNew = PPred - K * S * K.transpose();

    // Store
    X_ = XNew;
    P_ = PNew;

    // 10) Convert the *first/top block* of X_ into a SimilarityTransform
    //     (this is the “current, newly smoothed” state).
    SimilarityTransform currentFiltered;
    currentFiltered.A  = X_(0);
    currentFiltered.B  = X_(1);
    currentFiltered.TX = X_(2);
    currentFiltered.TY = X_(3);

    // 11) Also convert the *oldest block* into a SimilarityTransform and
    //     push it onto delayedOutput_ so that it will eventually get returned
    //     once it’s fully smoothed by N steps.
    {
        int oldestBlock = (lag_ - 1) * SINGLE_STATE_DIM;
        SimilarityTransform oldest;
        oldest.A  = X_(oldestBlock + 0);
        oldest.B  = X_(oldestBlock + 1);
        oldest.TX = X_(oldestBlock + 2);
        oldest.TY = X_(oldestBlock + 3);
        delayedOutput_.push_back(oldest);
    }

    // 12) Return the “fully smoothed” transform for the step that just fell off
    //     the end of our lag window, or identity if none yet.
    return out;
}

Eigen::VectorXd CameraMotionUKF_FixedLag::predictAugState(const Eigen::VectorXd &X_aug)
{
    Eigen::VectorXd Xpred = Eigen::VectorXd::Zero(augStateDim_);

    // SHIFT: X_{k-1} <- X_{k}, X_{k-2} <- X_{k-1}, ...
    // i.e. block i gets block i-1 from the old vector
    for (int block = lag_ - 1; block >= 1; block--) {
        int dst = block * SINGLE_STATE_DIM;
        int src = (block - 1) * SINGLE_STATE_DIM;
        Xpred.segment(dst, SINGLE_STATE_DIM) =
            X_aug.segment(src, SINGLE_STATE_DIM);
    }

    // For the *top* block (block=0), do your normal motion model
    {
        // old top block
        double A    = X_aug(0);
        double B    = X_aug(1);
        double TX   = X_aug(2);
        double TY   = X_aug(3);
        double vA   = X_aug(4);
        double vB   = X_aug(5);
        double vTX  = X_aug(6);
        double vTY  = X_aug(7);

        // Predict forward (dt=1)
        A  += vA;
        B  += vB;
        TX += vTX;
        TY += vTY;

        // Fill into Xpred(0..7)
        Xpred(0) = A;
        Xpred(1) = B;
        Xpred(2) = TX;
        Xpred(3) = TY;
        Xpred(4) = vA;
        Xpred(5) = vB;
        Xpred(6) = vTX;
        Xpred(7) = vTY;
    }
    return Xpred;
}

Eigen::VectorXd CameraMotionUKF_FixedLag::stateToMeasurement(const Eigen::VectorXd &X_aug)
{
    Eigen::VectorXd z(MEAS_DIM);
    z(0) = X_aug(0);
    z(1) = X_aug(1);
    z(2) = X_aug(2);
    z(3) = X_aug(3);
    return z;
}

void CameraMotionUKF_FixedLag::generateSigmaPoints(const Eigen::VectorXd &x,
                                                   const Eigen::MatrixXd &P,
                                                   std::vector<Eigen::VectorXd> &sigmaPoints)
{
    const int L = augStateDim_;
    sigmaPoints.resize(2*L + 1);

    // Compute sqrt of (L+lambda_) * P via Cholesky
    Eigen::MatrixXd A = ((L + lambda_) * P).llt().matrixL();

    sigmaPoints[0] = x;
    for (int i = 0; i < L; i++) {
        sigmaPoints[i + 1]       = x + A.col(i);
        sigmaPoints[i + 1 + L]   = x - A.col(i);
    }
}
