#pragma once

#include "Eigen/Dense"

void BoxProjection(Eigen::Matrix<double, Eigen::Dynamic, 1>const& y, Eigen::Matrix<double, Eigen::Dynamic, 1> const& a, Eigen::Matrix<double, Eigen::Dynamic, 1> const& b) {

	int i = -1;
	auto f = [&i, &a, &b](double x) { ++i; return x <= a(i) ? a(i) : (x >= b(i) ? b(i) : x); };
	const_cast<Eigen::Matrix<double, Eigen::Dynamic, 1>&>(y) = y.unaryExpr(f);
}