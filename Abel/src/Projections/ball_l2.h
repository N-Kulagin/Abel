#pragma once

#include "Eigen/Dense"

template <typename Derived>
void L2BallProjection_InPlace(Eigen::MatrixBase<Derived>const& vec, double radius = 1.0) {
	double norm_squared = vec.squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::MatrixBase<Derived>&>(vec) *= radius / sqrt(norm_squared); // cast to reference for block or matrix column inputs
}

template <typename Derived>
void L2BallProjection_InPlace(Eigen::ArrayBase<Derived>const& vec, double radius = 1.0) {
	double norm_squared = vec.matrix().squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::ArrayBase<Derived>&>(vec) *= radius / sqrt(norm_squared);
}

template <typename Derived1, typename Derived2>
void L2BallProjection(const Eigen::MatrixBase<Derived1>const& vec, Eigen::MatrixBase<Derived2> const& result, double radius = 1.0) {
	const_cast<Eigen::MatrixBase<Derived2>&>(result) = vec;
	double norm_squared = result.squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::MatrixBase<Derived2>&>(result) *= radius / sqrt(norm_squared);
}