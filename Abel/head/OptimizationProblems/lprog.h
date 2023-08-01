#pragma once

#include "lprog_Result.h"

// Solve linear programming problem c*x -> min subject to Ax = b, Bx <= d
lprog_Result lprog(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, 
	double tol = 1e-10, size_t max_iter = 100);

// Solve linear programming problem c*x -> min subject to Bx <= d
lprog_Result lprog(const Eigen::VectorXd& c, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol = 1e-10, size_t max_iter = 100);