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

	Eigen::MatrixXd x(dimension, dimension + 1); // simplex verticies
	Eigen::VectorXd centerOfMass(dimension); // center of mass
	Eigen::VectorXd reflection(dimension); // reflection point
	Eigen::VectorXd stretching(dimension); // stretching point
	Eigen::VectorXd contraction(dimension); // contraction point
	Eigen::ArrayXd f_vals(dimension + 1); // array of function values of the simplex

	if (hasStart){} // TODO: add a method to specify starting simplex
	else {
		x.setRandom();
		x *= randomCoeff; // spread the randomized verticies with coefficient randomCoeff
	}

	int bestVertexIndex = 0;
	int worstVertexIndex = 0;
	int secondWorstVertexIndex = -1;
	double val = f(x.col(0));
	double abs_min = val; // lowest function value
	double abs_max = val; // largest function value
	double f_reflection; // function value at reflection
	double f_stretching; // function value at stretching point
	double f_contraction; // function value at contraction point
	double f_secondWorst; // second largest function value
	double n = dimension; // dimension of the problem
	iter_counter = 0;

	for (size_t i = 0; i < n+1; i++)
	{
		val = f(x.col(i));
		if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
		if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
		f_vals(i) = val;
	}
	secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, n + 1);

	while (iter_counter < max_iter)
	{
		centerOfMass = (x.rowwise().sum() - x.col(worstVertexIndex)) / n;
		val = f(centerOfMass);
		error = sqrt(((f_vals - val).matrix().squaredNorm()) / (n + 1.0));
		if (error < tol) break;

		reflection = (1.0 + alpha) * centerOfMass - alpha * x.col(worstVertexIndex);
		f_reflection = f(reflection);
		f_secondWorst = f_vals(secondWorstVertexIndex);
		if (f_reflection <= abs_min) {
			stretching = (1.0 - gamma) * centerOfMass + gamma * reflection;
			f_stretching = f(stretching);
			if (f_stretching < abs_min) {
				x.col(worstVertexIndex) = stretching;
				abs_min = f_stretching;
				f_vals(worstVertexIndex) = f_stretching;
			}
			else {
				x.col(worstVertexIndex) = reflection;
				abs_min = f_reflection;
				f_vals(worstVertexIndex) = f_reflection;
			}
			abs_max = f_secondWorst;
			bestVertexIndex = worstVertexIndex;
			worstVertexIndex = secondWorstVertexIndex;
			secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, n + 1);
		}
		else if (f_secondWorst < f_reflection && f_reflection <= abs_max) {
			contraction = (1.0 - beta) * centerOfMass + beta * x.col(worstVertexIndex);
			f_contraction = f(contraction);
			x.col(worstVertexIndex) = contraction;
			f_vals(worstVertexIndex) = f_contraction;
			findMaxMin(f_vals, abs_max, abs_min, worstVertexIndex, bestVertexIndex, n + 1);
			secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, n + 1);
		}
		else if (abs_min < f_reflection && f_reflection <= f_secondWorst) {
			x.col(worstVertexIndex) = reflection;
			f_vals(worstVertexIndex) = f_reflection;
			findMaxMin(f_vals, abs_max, abs_min, worstVertexIndex, bestVertexIndex, n + 1);
			secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, n + 1);
		}
		else {
			for (size_t i = 0; i < n + 1; i++)
			{
				x.col(i) = (1.0 - delta) * x.col(bestVertexIndex) + delta * x.col(i);
			}
			val = f(x.col(0));
			abs_min = val;
			abs_max = val;
			for (size_t i = 0; i < n + 1; i++)
			{
				val = f(x.col(i));
				if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
				if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
				f_vals(i) = val;
			}
			secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, n + 1);
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

int MVNelderMead::findSecondWorst(const Eigen::ArrayXd& f_vals, double abs_max, int dimension)
{
	int secondWorstVertexIndex = -1;
	for (size_t i = 0; i < dimension; i++)
	{
		if (f_vals(i) != abs_max) {
			if (secondWorstVertexIndex == -1) secondWorstVertexIndex = i;
			else if (f_vals(i) > f_vals(secondWorstVertexIndex)) secondWorstVertexIndex = i; // find index of vertex with second largest function value
		}
	}
	return secondWorstVertexIndex;
}

void MVNelderMead::findMaxMin(const Eigen::ArrayXd& f_vals, double& abs_max, double& abs_min, int& worstVertexIndex, int& bestVertexIndex, int dimension)
{
	double val = f_vals(0);
	abs_max = val;
	abs_min = val;
	for (size_t i = 0; i < dimension; i++)
	{
		val = f_vals(i);
		if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
		if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
	}
}
