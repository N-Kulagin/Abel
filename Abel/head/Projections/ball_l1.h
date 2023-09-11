#pragma once

#include "Eigen/Dense"
#include "Projections/simplex.h"

template <typename Derived>
void L1BallProjection(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {
	// J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra
	// Efficient Projections onto the l1-Ball for Learning in High Dimensions
	// in Proceedings of the 25th international conference on Machine learning. ACM, 2008, pp. 272–279.
	// https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

	// minimize 1/2 ||y-x||_2^2 subject to ||y||_1 <= beta

	if (y.cols() > 1) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	Eigen::Matrix<double,Eigen::Dynamic,1> u = y.cwiseAbs();
	if (u.sum() <= beta) return;
	SimplexProjection(u, beta);
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([](double x) {return (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); });
	const_cast<Eigen::MatrixBase<Derived>&>(y) = u.array() * y.array();
}

template <typename Derived>
void L1BallProjection(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {
	for (size_t i = 0; i < y.cols(); i++)
	{
		L1BallProjection(y.col(i),beta);
	}
}