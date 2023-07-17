#pragma once

#include "Eigen/Dense"
#include "Projections/simplex.h"

template <typename Derived>
void L1BallProjection(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {
	if (y.cols() > 1) throw 1;
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