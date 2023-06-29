#pragma once

#include <Eigen/Core>

template <typename Derived>
void ProjL2Ball_InPlace(Eigen::MatrixBase<Derived>const& vec, double radius = 1.0) {
	double norm_squared = vec.squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::MatrixBase<Derived>&>(vec) *= radius / sqrt(norm_squared); // cast to reference for block or matrix column inputs
}

template <typename Derived>
void ProjL2Ball_InPlace(Eigen::ArrayBase<Derived>const& vec, double radius = 1.0) {
	double norm_squared = vec.matrix().squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::ArrayBase<Derived>&>(vec) *= radius / sqrt(norm_squared);
}

template <typename Derived1, typename Derived2>
void ProjL2Ball(const Eigen::MatrixBase<Derived1>const& vec, Eigen::MatrixBase<Derived2> const& result, double radius = 1.0) {
	const_cast<Eigen::MatrixBase<Derived2>&>(result) = const_cast<const Eigen::MatrixBase<Derived1>&>(vec);
	double norm_squared = result.squaredNorm();
	if (norm_squared <= radius) return;
	const_cast<Eigen::MatrixBase<Derived2>&>(result) *= radius / sqrt(norm_squared);
}