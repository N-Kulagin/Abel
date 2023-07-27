#pragma once

#include "Eigen/Dense"

class MVNumericalMethod {

protected:
	double tol;
	double error;
	int max_iter;
	int iter_counter;
	size_t dimension;
	Eigen::VectorXd result;
	bool was_run;

	MVNumericalMethod(size_t dimension, double tol = 0.01, int max_iter = 100, bool was_run = 0, int iter_counter = 0, double err = 0.0,
		const Eigen::VectorXd& res = Eigen::VectorXd());
	MVNumericalMethod& operator=(const MVNumericalMethod& nm);

public:
	virtual bool wasRun() const final;
	virtual size_t niter() const final;
	virtual void solve();
	virtual Eigen::VectorXd& getResult() const final;
	virtual double getError() const final;

};