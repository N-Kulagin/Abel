#include "pch.h"
#include "SingleVarOptimization/SVGoldenSection.h"

SVGoldenSection::SVGoldenSection(const std::function<double(double)>& f, double tol, double lb, double ub, size_t max_iter) : 
	f(f), lb(lb), ub(ub), SVNumericalMethod(tol, max_iter) {}

SVGoldenSection::SVGoldenSection(const SVGoldenSection& g) : f(g.f), lb(g.lb), ub(g.ub), SVNumericalMethod(g.tol, g.max_iter, g.iter_counter, g.result, g.error) {}

SVGoldenSection& SVGoldenSection::operator=(const SVGoldenSection& g)
{
	f = g.f;
	lb = g.lb;
	ub = g.ub;
	SVNumericalMethod::operator=(g);
	return *this;
}

void SVGoldenSection::setParams(double tol_, double lb_, double ub_, size_t max_iter_)
{
	tol = tol_;
	lb = lb_;
	ub = ub_;
	max_iter = max_iter_;
	iter_counter = 0;
	error = 0.0;
	result = 0.0;
}

void SVGoldenSection::solve()
{
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
	error = abs(b - a);
}

std::ostream& operator<<(std::ostream& out, const SVGoldenSection& g) {
	out << "Golden Section information:" << '\n'
		<< "Tolerance: " << g.tol << '\n'
		<< "Max iterations: " << g.max_iter << '\n'
		<< "Performed iterations: " << g.iter_counter << '\n'
		<< "LB/UB: " << g.lb << ' ' << g.ub;
	return out;
}


