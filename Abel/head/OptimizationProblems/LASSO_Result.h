#pragma once

struct LASSO_Result {
	Eigen::VectorXd result;
	double error;
	size_t niter;
	double L;

	LASSO_Result(const Eigen::VectorXd& x, double err, size_t iter, double lip);
};