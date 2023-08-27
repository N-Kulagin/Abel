#pragma once

class SVNumericalMethod {

protected:
	double tol;
	size_t max_iter;
	size_t iter_counter;
	double result;
	double error;

	SVNumericalMethod(double tol = 0.01, size_t max_iter = 100, size_t iter_counter = 0, double result = 0.0, double error = 0.0);
	SVNumericalMethod& operator=(const SVNumericalMethod& nm);

public:
	SVNumericalMethod() = delete;
	virtual size_t niter() const final;
	virtual void solve();
	virtual double getResult() const final;
	virtual double getError() const final;
};