#pragma once

#include "MultiVarOptimization/MVGradientDescent.h"
#include "LASSO_Result.h"

/*
struct LASSO_Result {
	Eigen::VectorXd result;
	double error;
	size_t niter;
	double L;
	
	LASSO_Result(const Eigen::VectorXd& x, double err, size_t iter, double lip): result(x),error(err),niter(iter),L(lip) {}
};
*/

// Solves LASSO problem, i. e. 0.5 * ||Ax-b||_2^2 + beta * ||x||_1 -> min
// Inputs are matrix A, vector b, tolerance (measure of how well the problem is solved) and maximum number of iterations
// Tolerance should be between 0 and 0.1
// Can refine solution using least squares with respect to variables which are above the refine_level in absolute value
// Solution is stored in the output struct, it also contains Lipschitz constant L (largest singular value of A^T * A)
LASSO_Result LASSO(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& start_point, const double beta = 1.0, const double tol = 1e-7,
	const bool refine_sol = true, const double refine_level = 1e-10, const int max_iter = 500);
/*
{
	
	if (A.rows() != b.rows() || start_point.rows() != A.cols()) throw 1;

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

	// L = ||A^T * A||_2 = ||A||_2^2 - lipschitz constant
	double L = pow(A.operatorNorm(), 2.0);

	MVGradientDescent gr(f, f_grad, A.cols(), g, tol, 1.0 / L, max_iter);
	gr.setStart(start_point);
	gr.setProx(prox);
	gr.toggleConstStep();
	gr.toggleConvex();
	gr.solve();

	LASSO_Result out(gr.getResult(), gr.getError(), gr.niter(), L);

	// solve regular least squares with respect to non-zero variables
	if (refine_sol) {
		Eigen::VectorXi indices(out.result.rows()); // vector of indices corresponding to non-zero components
		size_t index_counter = 0;

		for (size_t i = 0; i < indices.rows(); i++) // if result is bigger in abs than refine_level it's counted as non-zero
		{
			if (abs(out.result(i)) >= std::max(1e-15,refine_level)) { indices(index_counter) = i; ++index_counter; }
		}

		Eigen::MatrixXd B(A.rows(), index_counter); // create a submatrix related to non-zero components of the solution
		for (size_t i = 0; i < index_counter; i++)
		{
			B.col(i) = A.col(indices(i));
		}

		Eigen::VectorXd y = (B.transpose() * B).ldlt().solve(B.transpose() * b); // solve least squares
		for (size_t i = 0; i < index_counter; i++)
		{
			out.result(indices(i)) = y(i); // replace with results of the least squares
		}
	}
	return out;
}
*/