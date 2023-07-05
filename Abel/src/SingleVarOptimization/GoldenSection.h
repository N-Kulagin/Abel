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
	SVGoldenSection(const std::function<double(double)>& f, double tol = 0.01, double lb = -1e+5, double ub = 1e+5, size_t max_iter = 100) :
		f(f), lb(lb), ub(ub), SVNumericalMethod(tol, max_iter) {}

	SVGoldenSection(const SVGoldenSection& g) : f(g.f), lb(g.lb), ub(g.ub), SVNumericalMethod(g.tol, g.max_iter, g.wasRun, g.iter_counter, g.result) {}

	SVGoldenSection& operator=(const SVGoldenSection& g) {
		f = g.f;
		lb = g.lb;
		ub = g.ub;
		tol = g.tol;
		max_iter = g.max_iter;
		iter_counter = g.iter_counter;
		wasRun = g.wasRun;
		result = g.result;
		return *this;
	}

	friend std::ostream& operator<<(std::ostream&, const SVGoldenSection&); // is friend necessary??

	void setParams(double tol_ = 0.01, double lb_ = -1e+5, double ub_ = 1e+5, size_t max_iter_ = 100) {
		tol = tol_;
		lb = lb_;
		ub = ub_;
		max_iter = max_iter_;
		iter_counter = 0;
		wasRun = false;
		result = 0.0;
	}

	void solve() override {

		iter_counter = 0;

		double a = lb;
		double b = ub;
		double gr = (sqrt(5) - 1.0) / 2.0;

		double d = gr * (b - a);
		double x1 = a + d;
		double x2 = b - d;
		double y1 = f(x1);
		double y2 = f(x2);

		while (abs(b - a) > tol && iter_counter < max_iter) {
			++iter_counter;

			if (y1 < y2) {
				a = x2;
				x2 = x1;
				y2 = y1;

				d = gr * (b - a);
				x1 = a + d;
				y1 = f(x1);
			}
			else {
				b = x1;
				x1 = x2;
				y1 = y2;

				d = gr * (b - a);
				x2 = b - d;
				y2 = f(x2);
			}
		}
		result = (a + b) / 2.0;
		wasRun = true;
	}

};

std::ostream& operator<<(std::ostream& out, const SVGoldenSection& g) {
	out << "Golden Section information:" << '\n'
		<< "Tolerance: " << g.tol << '\n'
		<< "Max iterations: " << g.max_iter << '\n'
		<< "Performed iterations: " << g.iter_counter << '\n'
		<< "LB/UB: " << g.lb << ' ' << g.ub;
	return out;
}