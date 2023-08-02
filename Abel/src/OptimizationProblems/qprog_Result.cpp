#include "pch.h"
#include "OptimizationProblems/qprog_Result.h"

qprog_Result::qprog_Result(const Eigen::VectorXd& result, const Eigen::VectorXd& dual_variables, double error, size_t niter, bool isDiverging, int z_num, int l_num) : 
 result(result), dual_variables(dual_variables), error(error), niter(niter), isDiverging(isDiverging), z_num(z_num), l_num(l_num) {}
