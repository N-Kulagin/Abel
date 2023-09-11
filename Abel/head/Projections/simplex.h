#pragma once

#include "Eigen/Dense"
#include <unordered_set>
#include <random>

template <typename Derived>
void SimplexProjection(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {

	// J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra
	// Efficient Projections onto the l1-Ball for Learning in High Dimensions
	// in Proceedings of the 25th international conference on Machine learning. ACM, 2008, pp. 272–279.
	// https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
	// (includes O(n) projection onto a simplex)

	// minimize 1/2 ||y-x||_2^2 subject to sum of y_i = beta, y >= 0

	if (y.cols() > 1) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);

	bool inside_simplex = true;
	bool satisfy_equality = false;
	if (abs(y.sum() - beta) <= 1e-7) satisfy_equality = true;
	for (size_t i = 0; i < y.rows(); i++)
	{
		if (y(i) < 0) { inside_simplex = false; break; }
	}
	if (inside_simplex && satisfy_equality) return;


	double s = 0.0;
	double rho = 0.0;
	double d_rho = 0.0;
	double d_s = 0.0;
	size_t k = 0;
	double theta = 0.0;
	std::unordered_set<size_t> U;
	std::unordered_set<size_t> G;
	std::unordered_set<size_t> L;

	for (int i = 0; i < y.rows(); i++)
	{
		U.insert(i);
	}
	while (U.size() != 0)
	{
		std::mt19937 gen(std::random_device{}());
		std::sample(U.begin(), U.end(), &k, 1, gen); // since c++17
		auto it = U.begin();
		while (it != U.end())
		{
			if (y(*it) >= y(k)) {
				G.insert(*it);
				d_rho += 1.0;
				d_s += y(*it);
			}
			else L.insert(*it);
			++it;
		}
		if (s + d_s - (rho + d_rho) * y(k) < beta) {
			s += d_s;
			rho += d_rho;
			U = std::move(L);
		}
		else {
			U = std::move(G);
			U.erase(k);
		}
		d_rho = 0.0;
		d_s = 0.0;
		G.clear();
		L.clear();
	}
	theta = (s - beta) / rho;
	const_cast<Eigen::MatrixBase<Derived>&>(y) = y.unaryExpr([&theta](double x) {return std::max(0.0, x - theta); });
}

template <typename Derived>
void SimplexProjection(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {
	for (int i = 0; i < y.cols(); i++)
	{
		SimplexProjection(y.col(i), beta);
	}
}