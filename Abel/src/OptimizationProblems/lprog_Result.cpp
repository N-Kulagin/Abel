#include "pch.h"
#include "OptimizationProblems/lprog_Result.h"

lprog_Result::lprog_Result(const Eigen::VectorXd& result, const Eigen::VectorXd& dual_variables, double error, size_t niter, bool isDiverging, int z_num, int s_num) :
	result(result), dual_variables(dual_variables), error(error), niter(niter), isDiverging(isDiverging), z_num(z_num), s_num(s_num) {}