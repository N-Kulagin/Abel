#pragma once

#include "Eigen/Dense"

template <typename Derived>
void OrthantProjection(Eigen::MatrixBase<Derived> const& x) {
	const_cast<Eigen::MatrixBase<Derived>&>(x) = x.unaryExpr([](double x) { return x >= 0 ? x : 0.0; });
}