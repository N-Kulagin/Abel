#pragma once

#include "Eigen/Dense"

template <typename Derived>
void PSDConeProjection(const Eigen::MatrixBase<Derived>& X) {
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
	eig.compute(X);
	Eigen::VectorXd eigvals = eig.eigenvalues().unaryExpr([](double x) { return (x >= 0.0) ? x : 0.0; });
	const_cast<Eigen::MatrixBase<Derived>&>(X) = eig.eigenvectors() * eigvals.asDiagonal() * eig.eigenvectors().transpose();;
}