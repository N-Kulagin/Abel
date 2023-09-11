#include "pch.h"
#include "MultiVarOptimization/MVNelderMead.h"

MVNelderMead::MVNelderMead(std::function<double(const Eigen::VectorXd& x)> f, size_t dimension, double tol, int max_iter, bool hasLog) : 
	f(f), MVNumericalMethod(dimension, tol, max_iter, hasLog) {
	if (hasLog) {
		lg = AbelLogger(3);
	}
}

MVNelderMead::MVNelderMead(const MVNelderMead& nm) : f(nm.f), x(nm.x), MVNumericalMethod(nm.dimension, nm.tol, nm.max_iter, nm.hasLog, nm.iter_counter, nm.error,
	nm.result, nm.starting_point, nm.hasStart, nm.lg) {}

MVNelderMead MVNelderMead::operator=(const MVNelderMead& nm)
{
	f = nm.f;
	x = nm.x;
	MVNumericalMethod::operator=(nm);
	return *this;
}

void MVNelderMead::solve() noexcept
{

	if (hasStart) {
		solve_(*x);
	}
	else {
		Eigen::MatrixXd x(dimension, dimension + 1); // simplex verticies
		x.setRandom();
		x *= randomCoeff; // spread the randomized verticies with coefficient randomCoeff
		solve_(x);
	}
	hasStart = false;
}

void MVNelderMead::solve_(Eigen::MatrixXd& x)
{
	// Nelder-Mead minimization algorithm
	// https://www.scilab.org/sites/default/files/neldermead.pdf
	// Above mentioned paper uses following parameter convention (in paper vs this implmenetation):
	// rho = alpha, chi = gamma, gamma-bar = beta, sigma = delta
	// Stopping criterion used:  error = sqrt( 1/(n+1) * sum_i (f(x^i) - f(x^S))^2 ), where x^S - center of mass

	Eigen::VectorXd centerOfMass(dimension); // center of mass
	Eigen::VectorXd reflection(dimension); // reflection point
	Eigen::VectorXd stretching(dimension); // stretching point
	Eigen::VectorXd contraction(dimension); // contraction point
	Eigen::ArrayXd f_vals(dimension + 1); // array of function values of the simplex

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
	double n = (double)dimension; // dimension of the problem
	bool flag = false; // used for optimization purposes
	iter_counter = 0;

	for (int i = 0; i < n + 1; i++)
	{
		val = f(x.col(i));
		if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
		if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
		f_vals(i) = val;
	}
	secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, (int)(n + 1));

	while (iter_counter < max_iter)
	{
		centerOfMass = (x.rowwise().sum() - x.col(worstVertexIndex)) / n; // compute center of mass
		val = f(centerOfMass);
		error = sqrt(((f_vals - val).matrix().squaredNorm()) / (n + 1.0));
		if (error < tol) break;

		reflection = (1.0 + alpha) * centerOfMass - alpha * x.col(worstVertexIndex); // reflection step
		f_reflection = f(reflection);
		f_secondWorst = f_vals(secondWorstVertexIndex);
		if (f_reflection <= abs_min) {
			stretching = (1.0 + alpha * gamma) * centerOfMass - (alpha * gamma) * reflection; // expansion step
			f_stretching = f(stretching);
			if (f_stretching < f_reflection) {
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
			flag = true; // flag indicates that this step was used, hence we can optimize and not search for max and min
		}
		else if (abs_min < f_reflection && f_reflection <= f_secondWorst) {
			x.col(worstVertexIndex) = reflection;
			f_vals(worstVertexIndex) = f_reflection;
		}
		else if (f_secondWorst < f_reflection && f_reflection <= abs_max) {
			contraction = (1.0 + alpha * beta) * centerOfMass - alpha * beta * x.col(worstVertexIndex); // outside contraction
			f_contraction = f(contraction);
			if (f_contraction < f_reflection) {
				x.col(worstVertexIndex) = contraction;
				f_vals(worstVertexIndex) = f_contraction;
			}
			else {
				shrinkAndRecalculate(x, f_vals, abs_max, abs_min, worstVertexIndex, bestVertexIndex, (int)(n + 1)); // shrink simplex and find largest and smallest value at new verticies
				secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, (int)(n + 1));
			}
		}
		else {
			contraction = (1.0 - beta) * centerOfMass + beta * x.col(worstVertexIndex); // inside contraction
			f_contraction = f(contraction);
			if (f_contraction < abs_max) {
				x.col(worstVertexIndex) = contraction;
				f_vals(worstVertexIndex) = f_contraction;
			}
			else {
				shrinkAndRecalculate(x, f_vals, abs_max, abs_min, worstVertexIndex, bestVertexIndex, (int)(n + 1));
			}
		}
		if (!flag) { // find new largest, second largest and smallest function value at verticies
			findMaxMin(f_vals, abs_max, abs_min, worstVertexIndex, bestVertexIndex, (int)(n + 1));
		}
		if (hasLog) {
			lg.record(error, 0);
			lg.record(abs_min, 1);
			lg.record(abs_max, 2);
		}
		secondWorstVertexIndex = findSecondWorst(f_vals, abs_max, (int)(n + 1));
		flag = false;
		++iter_counter;
	}
	result = x.col(bestVertexIndex);
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
}

void MVNelderMead::setStart(Eigen::MatrixXd& simplex)
{
	if (simplex.rows() == dimension && simplex.cols() == dimension + 1) {
		x = &simplex;
		hasStart = true;
	}
	else throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
}

void MVNelderMead::printLogs() const
{
	lg.print("MVNelderMead", { "Error", "ObjectiveMin", "ObjectiveMax" });
}

int MVNelderMead::findSecondWorst(const Eigen::ArrayXd& f_vals, double abs_max, int dimension)
{
	int secondWorstVertexIndex = -1;
	for (int i = 0; i < dimension; i++)
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
	for (int i = 0; i < dimension; i++)
	{
		val = f_vals(i);
		if (val <= abs_min) { abs_min = val; bestVertexIndex = i; }
		if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
	}
}

void MVNelderMead::shrinkAndRecalculate(Eigen::MatrixXd& x, Eigen::ArrayXd& f_vals, double& abs_max, double& abs_min, int& worstVertexIndex, int& bestVertexIndex, int dimension)
{
	for (size_t i = 0; i < dimension; i++)
	{
		x.col(i) = (1.0 - delta) * x.col(bestVertexIndex) + delta * x.col(i);
	}
	double val = f(x.col(0));
	abs_min = val;
	abs_max = val;
	for (int i = 0; i < dimension; i++)
	{
		val = f(x.col(i));
		if (val <= abs_min) { abs_min = val; bestVertexIndex = i; } // compute values of the function at verticies and determine index of vertex with largest/smallest value
		if (val >= abs_max) { abs_max = val; worstVertexIndex = i; }
		f_vals(i) = val;
	}
}
