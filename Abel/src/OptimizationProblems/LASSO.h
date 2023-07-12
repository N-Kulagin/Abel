#pragma once

#include "Eigen/Dense"
#include "MultiVarOptimization/MVGradientDescent.h"
#include "Prox/prox_l1.h"

// Solves LASSO problem, i. e. 0.5 * ||Ax-b||_2^2 + beta * ||x||_1 -> min
// Inputs are matrix A, vector b, tolerance (measure of how well the problem is solved) and maximum number of iterations
// Tolerance should be between 0 and 0.1
// Solution is stored in the "result", LASSO returns an array [solution_error,tolerance,number_of_iterations,max_iter,L]
// where L is square of the 2-norm (operator norm) of A, i. e. square of largest singular value
Eigen::Matrix<double, 5, 1> LASSO(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& result, double beta = 1.0, double tol = 1e-7, int max_iter = 500) {

	result.setRandom();
	double beta_safe = std::max(0.0, beta);

	auto f = [&A, &b](const Eigen::VectorXd& x) {
		return 0.5 * (A * x - b).squaredNorm();
	};
	auto g = [&beta_safe](const Eigen::VectorXd& x) {
		return beta_safe * x.lpNorm<1>();
	};
	auto f_grad = [&A, &b](Eigen::VectorXd& grad, const Eigen::VectorXd& x) {
		grad = A.transpose() * (A * x - b);
	};

	auto prox = [&beta_safe](Eigen::VectorXd& x, double alpha) {
		L1Prox(x, beta_safe * alpha);
	};

	// L = ||A^T * A||_2 = ||A||_2^2
	double L = pow(A.operatorNorm(), 2.0);

	MVGradientDescent gr(f, f_grad, A.cols(), g, tol, 1.0 / L, max_iter, result);
	gr.setProx(prox);
	gr.toggleConstStep();
	gr.toggleConvex();
	gr.solve();
	result = gr.getResult();

	Eigen::Matrix<double, 5, 1> params;
	params << gr.getError(), tol, gr.niter(), max_iter, L;
	return params;

}