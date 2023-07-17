#pragma once

#include <iostream>
#include <functional>
#include "Base/SVNumericalMethod.h"

class SVGoldenSection : public SVNumericalMethod {

private:
	std::function<double(double)> f;
	double lb;
	double ub;
	const double gr = (sqrt(5.0) - 1.0) / 2.0;

public:
	SVGoldenSection(const std::function<double(double)>& f, double tol = 0.01, double lb = -1e+5, double ub = 1e+5, size_t max_iter = 100);

	SVGoldenSection(const SVGoldenSection& g);

	SVGoldenSection& operator=(const SVGoldenSection& g);

	friend std::ostream& operator<<(std::ostream&, const SVGoldenSection&);

	void setParams(double tol_ = 0.01, double lb_ = -1e+5, double ub_ = 1e+5, size_t max_iter_ = 100);

	void solve() override;

};