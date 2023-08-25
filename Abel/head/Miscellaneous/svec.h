#pragma once
#include "Eigen/Dense"

// symmetric vectorization function, given symmetric n by n matrix A, produces a vector of dimension n*(n+1)/2
// where the components are (a_11, sqrt(2)a_12, sqrt(2)a_13, ..., sqrt(2)a_1n, a_22, sqrt(2)a_23, ..., a_nn)
// this is an isometry between space of symmetric matrices and vectors, it presevers standard inner product trace vs dot product
template <typename Derived>
void svec(const Eigen::MatrixXd& A, const Eigen::MatrixBase<Derived>& res) {
	int i = 0;
	int j = -1;
	int cols = (int)A.cols();
	const_cast<Eigen::MatrixBase<Derived>&>(res) = res.unaryExpr([&A, &i, &j, cols](double x) {
		++j;
		if (j == cols) {
			++i;
			j = i;
		}
		return ((i == j) ? A(i, j) : sqrt(2) * A(i, j));
		});
}