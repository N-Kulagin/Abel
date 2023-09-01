#pragma once

#include "Base/MVNumericalMethod.h"
#include "Base/SVNumericalMethod.h"
#include "IPM/MVIntPointLP.h"
#include "IPM/MVIntPointQP.h"
#include "IPM/MVIntPointSDP.h"
#include "Miscellaneous/AbelLogger.h"
#include "Miscellaneous/svec.h"
#include "Miscellaneous/smat.h"
#include "MultiVarOptimization/MVGradientDescent.h"
#include "MultiVarOptimization/MVNewton.h"
#include "MultiVarOptimization/MVNelderMead.h"
#include "OptimizationProblems/LASSO_Result.h"
#include "OptimizationProblems/LASSO.h"
#include "OptimizationProblems/lprog.h"
#include "OptimizationProblems/lprog_Result.h"
#include "OptimizationProblems/qprog.h"
#include "OptimizationProblems/qprog_Result.h"
#include "Projections/affine.h"
#include "Projections/ball_l1.h"
#include "Projections/ball_l2.h"
#include "Projections/ball_linf.h"
#include "Projections/box.h"
#include "Projections/orthant.h"
#include "Projections/psd_cone.h"
#include "Projections/simplex.h"
#include "Prox/prox_l1.h"
#include "Prox/prox_l2.h"
#include "Prox/prox_linf.h"
#include "RootFinding/SVNewton.h"
#include "RootFinding/SVNewtonGlobal.h"
#include "SingleVarOptimization/SVGoldenSection.h"

#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <chrono>