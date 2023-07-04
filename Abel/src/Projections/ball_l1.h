#pragma once

#include "Eigen/Dense"
#include "Projections/simplex.h"

template <typename Derived>
void L1BallProjection_Linear(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {
	Eigen::MatrixXd u = y.cwiseAbs();
	SimplexProjection_Linear(u, beta);
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([](double x) {return (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); });
	const_cast<Eigen::MatrixBase<Derived>&>(y) = u.array() * y.array();
}