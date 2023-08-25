#pragma once

#include "Eigen/Dense"

template <typename D1, typename D2, typename D3>
void AffineProjection(Eigen::MatrixBase<D1> const& A, Eigen::MatrixBase<D2> const& b, Eigen::MatrixBase<D3> const& input) {
	if (A.rows() > A.cols()) throw 1;
	const_cast<Eigen::MatrixBase<D3>&>(input) -= A.transpose() * (A * A.transpose()).llt().solve(A * input - b);
}