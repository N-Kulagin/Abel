#include "pch.h"
#include "Base/SVNumericalMethod.h"

SVNumericalMethod::SVNumericalMethod(double tol, size_t max_iter, size_t iter_counter, double result, double error) :
	tol(tol), max_iter(max_iter), iter_counter(iter_counter), result(result), error(error) {}

SVNumericalMethod& SVNumericalMethod::operator=(const SVNumericalMethod& nm)
{
	tol = nm.tol;
	max_iter = nm.max_iter;
	iter_counter = nm.iter_counter;
	error = nm.error;
	result = nm.result;
	return *this;
}

size_t SVNumericalMethod::niter() const
{
	return iter_counter;
}

void SVNumericalMethod::solve()
{
}

double SVNumericalMethod::getResult() const
{
	return result;
}

double SVNumericalMethod::getError() const
{
	return error;
}
