#pragma once
#include "Eigen/Dense"

// inverse of symmetric vectorization (svec), turns a vector into symmetric matrix
inline Eigen::MatrixXd smat(const Eigen::VectorXd& x, int n) {
	int el_count = 0;
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = i; j < n; j++)
		{
			A(i, j) = (i == j) ? x(el_count) : x(el_count) / sqrt(2);
			A(j, i) = A(i, j);
			++el_count;
		}
	}
	return A;
}