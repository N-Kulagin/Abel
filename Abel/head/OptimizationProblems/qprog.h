#pragma once

#include "OptimizationProblems/qprog_Result.h"

// Solve 0.5 * (x^T * G * x) + c^T*x subject to Ax=b, Bx<=d
qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d,
	double tol = 1e-10, size_t max_iter = 100);

qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d,
	double tol = 1e-10, size_t max_iter = 100);