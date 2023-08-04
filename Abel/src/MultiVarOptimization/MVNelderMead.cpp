#include "pch.h"
#include "MultiVarOptimization/MVNelderMead.h"

MVNelderMead::MVNelderMead(std::function<double(const Eigen::VectorXd& x)> f, size_t dimension, double tol, int max_iter) : f(f), MVNumericalMethod(dimension, tol, max_iter) {}

MVNelderMead::MVNelderMead(const MVNelderMead& nm) : f(nm.f), MVNumericalMethod(nm.dimension, nm.tol, nm.max_iter, nm.was_run, nm.iter_counter, nm.error,
	nm.result, nm.starting_point, nm.hasStart) {}

MVNelderMead MVNelderMead::operator=(const MVNelderMead& nm)
{
	f = nm.f;
	MVNumericalMethod::operator=(nm);
	return *this;
}

void MVNelderMead::solve() noexcept
{
	if (was_run) return;
	// TODO: this is a naive implementation, the algorithm should not evaluate *all* function values at each iterations
	// also smarter error computation needs to be done
	// http://www.scholarpedia.org/article/Nelder-Mead_algorithm

	Eigen::MatrixXd x(dimension, dimension + 1); // simplex verticies
	Eigen::VectorXd centerOfMass(dimension); // center of mass
	Eigen::VectorXd reflection(dimension); // reflection point
	Eigen::VectorXd stretching(dimension); // stretching point
	Eigen::VectorXd contraction(dimension); // contraction point
	Eigen::ArrayXd f_vals(dimension + 1); // array of function values of the simplex

	x.setRandom();
	x *= randomCoeff; // spread the randomized verticies with coefficient randomCoeff

	int bestVertexIndex = 0;
	int worstVertexIndex;
	int secondWorstVertexIndex;
	double val = f(x.col(0));
	double abs_min = val; // lowest function value
	double abs_max = val; // largest function value
	double f_reflection; // function value at reflection
	double f_secondWorst; // second largest function value
	double n = dimension; // dimension of the problem
	iter_counter = 0;

	while (iter_counter < max_iter)
	{
		bestVertexIndex = 0;
		worstVertexIndex = 0;
		secondWorstVertexIndex = -1;
		abs_min = f(x.col(0));
		abs_max = abs_min;

		for (size_t i = 0; i < n + 1; i++)
		{
			val = f(x.col(i));
			if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
			if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
			f_vals(i) = val;
		}
		for (size_t i = 0; i < n + 1; i++)
		{
			if (f_vals(i) != abs_max) {
				if (secondWorstVertexIndex == -1) secondWorstVertexIndex = i;
				else if (f_vals(i) > f_vals(secondWorstVertexIndex)) secondWorstVertexIndex = i; // find index of vertex with second largest function value
			}
		}
		centerOfMass = (x.rowwise().sum() - x.col(worstVertexIndex)) / n;
		val = f(centerOfMass);
		error = sqrt(((f_vals - val).matrix().squaredNorm()) / (n + 1.0)); // can be done more efficiently
		if (error < tol) break;

		reflection = (1.0 + alpha) * centerOfMass - alpha * x.col(worstVertexIndex);
		f_reflection = f(reflection);
		f_secondWorst = f(x.col(secondWorstVertexIndex));
		if (f_reflection <= abs_min) {
			stretching = (1.0 - gamma) * centerOfMass + gamma * reflection;
			if (f(stretching) < abs_min) {
				x.col(worstVertexIndex) = stretching;
			}
			else {
				x.col(worstVertexIndex) = reflection;
			}
		}
		else if (f_secondWorst < f_reflection && f_reflection <= abs_max) {
			contraction = (1.0 - beta) * centerOfMass + beta * x.col(worstVertexIndex);
			x.col(worstVertexIndex) = contraction;
		}
		else if (abs_min < f_reflection && f_reflection <= f_secondWorst) {
			x.col(secondWorstVertexIndex) = reflection;
		}
		else {
			for (size_t i = 0; i < n + 1; i++)
			{
				x.col(i) = (1.0 - delta) * x.col(bestVertexIndex) + delta * x.col(i);
			}
		}
		++iter_counter;
	}
	result = x.col(bestVertexIndex);
	was_run = true;
}

void MVNelderMead::setParams(double tol_, int max_iter_, double randomCoeff_, double alpha_, double beta_, double gamma_, double delta_)
{
	tol = std::max(1e-15, tol_);
	max_iter = std::max(2, max_iter_);
	randomCoeff = std::max(1e-14, abs(randomCoeff_));
	alpha = std::max(0.001, alpha_);
	beta = std::min(0.999, std::max(beta_, 0.001));
	gamma = std::max({ gamma_, alpha_+0.001, 1.001 });
	delta = std::min(0.999, std::max(delta_, 0.001));
	was_run = false;
}

