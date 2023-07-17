#include "pch.h"
#include "OptimizationProblems/LASSO_Result.h"

LASSO_Result::LASSO_Result(const Eigen::VectorXd& x, double err, size_t iter, double lip): result(x), error(err), niter(iter), L(lip) {}