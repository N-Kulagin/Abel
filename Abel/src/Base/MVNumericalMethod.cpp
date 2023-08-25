#include "pch.h"
#include "Base/MVNumericalMethod.h"

MVNumericalMethod::MVNumericalMethod(size_t dimension, double tol, size_t max_iter, int iter_counter, double err,
	const Eigen::VectorXd& res, const Eigen::VectorXd& starting_point, bool hasStart) :
	tol(std::min(std::max(1e-14, tol), 0.1)), max_iter(std::max((size_t)2, max_iter)), iter_counter(iter_counter), error(err), result(res), dimension(dimension),
	starting_point(starting_point), hasStart(hasStart)
{
	if (result.size() == 0) {
		result.resize(dimension);
		result.setConstant(1.0);
	}
}

MVNumericalMethod& MVNumericalMethod::operator=(const MVNumericalMethod& nm)
{
	tol = nm.tol;
	error = nm.error;
	max_iter = nm.max_iter;
	iter_counter = nm.iter_counter;
	dimension = nm.dimension;
	result = nm.result;
	hasStart = nm.hasStart;
	starting_point = nm.starting_point;
	return *this;
}

size_t MVNumericalMethod::niter() const
{
	return iter_counter;
}

void MVNumericalMethod::solve() noexcept
{
}

Eigen::VectorXd& MVNumericalMethod::getResult() const
{
	return const_cast<Eigen::VectorXd&>(result);
}

double MVNumericalMethod::getError() const
{
	return error;
}
