#pragma once

class SVNumericalMethod {

protected:
	double tol;
	size_t max_iter;
	size_t iter_counter;
	double result;
	bool wasRun;

	SVNumericalMethod(double tol = 0.01, size_t max_iter = 100, bool wasRun = 0, size_t iter_counter = 0, double result = 0.0) :
		tol(tol), max_iter(max_iter), iter_counter(iter_counter), wasRun(wasRun), result(result) {}

public:
	virtual bool was_run() const final { return wasRun; };
	virtual size_t niter() const final { return iter_counter; };
	virtual void solve() {};
	virtual double getResult() const final { return result; }
};