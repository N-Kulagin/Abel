#pragma once

#include "Eigen/Dense"

template <typename Derived>
void LInfBallProjection(Eigen::MatrixBase<Derived> const& y, double beta = 1.0) {

	if (y.cols() > 1) throw 1;
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([&beta](double x) {return abs(x) > beta ? beta * copysign(1.0, x) : x; });
}

template <typename Derived>
void LInfBallProjection(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {

	for (int i = 0; i < y.cols(); i++)
	{
		LInfBallProjection(y.col(i), beta);
	}
}