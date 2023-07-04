#pragma once

#include <functional>

double GoldenSection(std::function<double(double)> f, double tol = 0.01, double lb = -1e+5, double ub = 2e+5, int iter_limit = 1000) {

	double a = lb;
	double b = ub;
	double gr = (sqrt(5) - 1.0) / 2.0;

	double d = gr * (b - a);
	double x1 = a + d;
	double x2 = b - d;
	double y1 = f(x1);
	double y2 = f(x2);

	int iter_counter = 0;

	while (abs(b - a) > tol && iter_counter < iter_limit) {
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
	return (a + b) / 2.0;

}