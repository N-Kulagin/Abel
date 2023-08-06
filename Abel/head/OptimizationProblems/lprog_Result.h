#pragma once

struct lprog_Result {
	Eigen::VectorXd result;
	Eigen::VectorXd dual_variables;
	double error;
	size_t niter;
	bool isDiverging;
	size_t z_num; // number of dual variables for equality constraints
	size_t s_num; // number of dual variables for inequalities

	lprog_Result(const Eigen::VectorXd& result, const Eigen::VectorXd& dual_variables, double error, size_t niter, bool isDiverging, size_t z_num, size_t s_num);
};