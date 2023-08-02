#pragma once

struct qprog_Result {
	Eigen::VectorXd result;
	Eigen::VectorXd dual_variables;
	double error;
	size_t niter;
	bool isDiverging;
	int z_num; // number of dual variables for equality constraints
	int l_num; // number of dual variables for inequalities

	qprog_Result(const Eigen::VectorXd& result, const Eigen::VectorXd& dual_variables, double error, size_t niter, bool isDiverging, int z_num, int l_num);
};