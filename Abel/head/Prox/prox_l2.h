#pragma once

#include "Eigen/Dense"
#include <Projections/ball_l2.h>

template <typename Derived>
void L2Prox(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {

	// prox operator for g(x) = beta * ||x||_2 (using Moreau decomposition theorem)

	if (y.cols() > 1) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);

	if (y.norm() <= beta) {
		const_cast<Eigen::MatrixBase<Derived>&>(y) = const_cast<Eigen::MatrixBase<Derived>&>(y).setConstant(0.0);
		return;
	}
	Eigen::MatrixXd u = y;
	L2BallProjection_InPlace(u, beta);
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y - u;
}

template <typename Derived>
void L2Prox(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {
	for (size_t i = 0; i < y.cols(); i++)
	{
		L2Prox(y.col(i), beta);
	}
}