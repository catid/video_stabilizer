#pragma once

#include "constants.h"

#include <Halide.h>
using namespace Halide;


//------------------------------------------------------------------------------
// Conventional Variables

// Typically used for image coordinates
extern Var x, y, c;

// Typically used for tile()
extern Var yo, yi, yio, yii, xo, xi, xio, xii, tile_index;

// Typically used for rfactor()
extern Var u, v;
extern RVar rxo, rxi, rxio, rxii, ryo, ryi, ryio, ryii;
