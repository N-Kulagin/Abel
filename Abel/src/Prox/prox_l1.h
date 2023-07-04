#pragma once

#include "Eigen/Dense"

template <typename Derived>
void L1Prox(Eigen::MatrixBase<Derived> const& y, double beta = 1.0) {
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([&beta](double x) {return (abs(x) <= beta ? 0.0 : (x > beta ? x - beta : x + beta)); });
}