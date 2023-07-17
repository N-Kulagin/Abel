#pragma once

#include "MultiVarOptimization/MVGradientDescent.h"
#include "LASSO_Result.h"

// Solves LASSO problem, i. e. 0.5 * ||Ax-b||_2^2 + beta * ||x||_1 -> min
// Inputs are matrix A, vector b, tolerance (measure of how well the problem is solved) and maximum number of iterations
// Tolerance should be between 0 and 0.1
// Can refine solution using least squares with respect to variables which are above the refine_level in absolute value
// Solution is stored in the output struct, it also contains Lipschitz constant L (largest singular value of A^T * A)
LASSO_Result LASSO(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& start_point, const double beta = 1.0, const double tol = 1e-7,
	const bool refine_sol = true, const double refine_level = 1e-10, const int max_iter = 500);