#pragma once

#include "Eigen/Dense"

template <typename Derived>
void OrthantProjection(Eigen::MatrixBase<Derived> const& x) {
	// minimize 1/2 ||y-x||_2^2 subject to y >= 0
	const_cast<Eigen::MatrixBase<Derived>&>(x) = x.unaryExpr([](double x) { return x >= 0 ? x : 0.0; });
}