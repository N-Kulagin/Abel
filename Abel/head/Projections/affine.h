#pragma once

#include "Eigen/Dense"

template <typename D1, typename D2, typename D3>
void AffineProjection(Eigen::MatrixBase<D1> const& A, Eigen::MatrixBase<D2> const& b, Eigen::MatrixBase<D3> const& input) {
	// minimize 1/2 ||y-x||_2^2 over y subject to Ay=b (project vector x=input on affine subspace)

	if (A.rows() > A.cols()) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	const_cast<Eigen::MatrixBase<D3>&>(input) -= A.transpose() * (A * A.transpose()).llt().solve(A * input - b);
}