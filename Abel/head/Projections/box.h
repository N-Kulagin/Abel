#pragma once

#include "Eigen/Dense"

template <typename Derived>
void BoxProjection(Eigen::MatrixBase<Derived>const& y, Eigen::Matrix<double, Eigen::Dynamic, 1> const& a, Eigen::Matrix<double, Eigen::Dynamic, 1> const& b) {
	if (y.cols() != 1 || !(y.rows() == a.rows() && y.rows() == b.rows())) throw 1;
	int i = -1;
	auto f = [&i, &a, &b](double x) { ++i; return x <= a(i) ? a(i) : (x >= b(i) ? b(i) : x); };
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr(f);
}

template <typename Derived>
void BoxProjection(Eigen::MatrixBase<Derived>& y, Eigen::Matrix<double, Eigen::Dynamic, 1> const& a, Eigen::Matrix<double, Eigen::Dynamic, 1> const& b) {
	for (size_t i = 0; i < y.cols(); i++)
	{
		BoxProjection(y.col(i), a, b);
	}
}