#pragma once

#include "Eigen/Dense"

class MVNumericalMethod {

protected:
	double tol;
	int max_iter;
	int iter_counter;
	size_t dimension;
	Eigen::VectorXd result;
	bool was_run;

	MVNumericalMethod(size_t dimension, double tol = 0.01, int max_iter = 100, bool was_run = 0, int iter_counter = 0,
		const Eigen::VectorXd& res = Eigen::VectorXd()) :
		tol(std::min(std::max(1e-15, tol),0.1)), max_iter(std::max(2, max_iter)), iter_counter(iter_counter), was_run(was_run), result(res), dimension(dimension) {
		if (result.size() == 0) {
			result.resize(dimension);
			result.setConstant(1.0);
		}
	}

public:
	virtual bool wasRun() const final { return was_run; };
	virtual size_t niter() const final { return iter_counter; };
	virtual void solve() {};
	virtual Eigen::VectorXd& getResult() const final { return const_cast<Eigen::VectorXd&>(result); }

};