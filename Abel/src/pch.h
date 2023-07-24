#pragma once

#include "Base/MVNumericalMethod.h"
#include "Base/SVNumericalMethod.h"
#include "IPM/MVIntPointLP.h"
#include "MultiVarOptimization/MVGradientDescent.h"
#include "MultiVarOptimization/MVNewton.h"
#include "OptimizationProblems/LASSO_Result.h"
#include "OptimizationProblems/LASSO.h"
#include "Projections/affine.h"
#include "Projections/ball_l1.h"
#include "Projections/ball_l2.h"
#include "Projections/ball_linf.h"
#include "Projections/box.h"
#include "Projections/orthant.h"
#include "Projections/simplex.h"
#include "Prox/prox_l1.h"
#include "Prox/prox_l2.h"
#include "Prox/prox_linf.h"
#include "RootFinding/SVNewton.h"
#include "RootFinding/SVNewtonGlobal.h"
#include "SingleVarOptimization/SVGoldenSection.h"



#include <iostream>
#include <functional>
#include <algorithm>
#include <unordered_set>