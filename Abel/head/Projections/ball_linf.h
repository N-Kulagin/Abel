#pragma once

#include "Eigen/Dense"

template <typename Derived>
void LInfBallProjection(Eigen::MatrixBase<Derived> const& y, double beta = 1.0) {

	// minimize 1/2 ||y-x||_2^2 subject to ||y||_infinity <= beta

	if (y.cols() > 1) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([&beta](double x) {return abs(x) > beta ? beta * copysign(1.0, x) : x; });
}

template <typename Derived>
void LInfBallProjection(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {

	for (int i = 0; i < y.cols(); i++)
	{
		LInfBallProjection(y.col(i), beta);
	}
}