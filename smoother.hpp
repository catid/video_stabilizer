#pragma once

#include "imgproc.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>

class L1SmootherCenter
{
public:
    L1SmootherCenter(int lagBehind, int lagAhead, double lambda = 1.0);

    /**
     * Push a new measurement (for frame index = current total input - 1).
     * Returns an optional "finalized" transform if any frame is ready.
     * If none is ready, returns an identity transform and sets a flag.
     */
    bool update(const SimilarityTransform& meas,
                SimilarityTransform& outFinalized);

private:
    int m_lagBehind;          ///< how many past frames to include
    int m_lagAhead;           ///< how many future frames
    double m_lambda;          ///< smoothing strength
    int m_nextToFinalize;     ///< index of the next measurement to finalize

    std::vector<SimilarityTransform> m_measurements;
};