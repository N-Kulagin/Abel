#pragma once

#include "Eigen/Dense"
#include "Projections/ball_l1.h"

template <typename Derived>
void LInfProx(Eigen::MatrixBase<Derived> const& y, double beta = 1.0) {

	// prox operator for g(x) = beta * ||x||_infinity (using Moreau decomposition theorem)

	Eigen::MatrixXd u = y;
	L1BallProjection(u, beta);
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y - u;
}