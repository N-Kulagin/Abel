#pragma once

#include "SVNewton.h"

class SVNewtonGlobal : public SVNewton {

public:

	SVNewtonGlobal(const std::function<double(double)>& f, const std::function<double(double)>& f_prime,
		double tol = 0.01, size_t max_iter = 100, double starting_point = 1.0);

	void setParams(double tol_ = 0.01, size_t max_iter_ = 100, double starting_point_ = 1.0);

	void solve() override;
};