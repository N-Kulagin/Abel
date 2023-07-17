#include "pch.h"
#include "Base/MVNumericalMethod.h"

MVNumericalMethod::MVNumericalMethod(size_t dimension, double tol, int max_iter, bool was_run, int iter_counter, const Eigen::VectorXd& res) :
	tol(std::min(std::max(1e-15, tol), 0.1)), max_iter(std::max(2, max_iter)), iter_counter(iter_counter), was_run(was_run), result(res), dimension(dimension)
{
	if (result.size() == 0) {
		result.resize(dimension);
		result.setConstant(1.0);
	}
}

bool MVNumericalMethod::wasRun() const
{
	return was_run;
}

size_t MVNumericalMethod::niter() const
{
	return iter_counter;
}

void MVNumericalMethod::solve()
{
}

Eigen::VectorXd& MVNumericalMethod::getResult() const
{
	return const_cast<Eigen::VectorXd&>(result);
}
