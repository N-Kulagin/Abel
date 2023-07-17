#pragma once

#include <iostream>
#include <functional>
#include "Base/SVNumericalMethod.h"

class SVNewton : public SVNumericalMethod {

protected:
	std::function<double(double)> f;
	std::function<double(double)> f_prime;
	double heuristic;
	double starting_point;
	double multiplicity;

public:

	SVNewton(const std::function<double(double)>& f, const std::function<double(double)>& f_prime,
		double tol = 0.01, double heuristic = 0.05, size_t max_iter = 100, double starting_point = 1.0);

	SVNewton(const SVNewton& n);

	SVNewton& operator=(const SVNewton& n);

	friend std::ostream& operator<<(std::ostream&, const SVNewton&);

	virtual void setParams(double tol_ = 0.01, double heuristic_ = 0.05, size_t max_iter_ = 100, double starting_point_ = 1.0);

	double getMultiplicity();

	virtual void solve() override;
};