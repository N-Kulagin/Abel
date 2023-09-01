#pragma once

#include "Eigen/Dense"
#include "Miscellaneous/AbelLogger.h"

class MVNumericalMethod {

protected:
	size_t dimension;
	Eigen::VectorXd starting_point;
	double tol;
	double error;
	size_t max_iter;
	int iter_counter;
	Eigen::VectorXd result;
	bool hasStart;
	bool hasLog; // true if logging is enabled
	AbelLogger lg; // logger class

	MVNumericalMethod(size_t dimension, double tol = 0.01, size_t max_iter = 100, bool hasLog = false, int iter_counter = 0, double err = 0.0,
		const Eigen::VectorXd& res = Eigen::VectorXd(), const Eigen::VectorXd& starting_point = Eigen::VectorXd(), bool hasStart = false, const AbelLogger& lg = AbelLogger());
	MVNumericalMethod& operator=(const MVNumericalMethod& nm);

public:
	virtual size_t niter() const final;
	virtual void solve() noexcept;
	virtual Eigen::VectorXd& getResult() const final;
	virtual double getError() const final;
	virtual void printLogs() const;

};