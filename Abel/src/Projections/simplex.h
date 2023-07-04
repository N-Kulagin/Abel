#pragma once

#include "Eigen/Dense"
#include <unordered_set>
#include <random>

template <typename Derived>
void SimplexProjection_Linear(Eigen::MatrixBase<Derived>const& y, double beta = 1.0) {

	if (y.cols() > 1) throw 1;

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
void SimplexProjection_Linear(Eigen::MatrixBase<Derived>& y, double beta = 1.0) {
	for (int i = 0; i < y.cols(); i++)
	{
		SimplexProjection_Linear(y.col(i), beta);
	}
}