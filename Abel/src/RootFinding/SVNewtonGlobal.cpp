#include "pch.h"
#include "RootFinding/SVNewtonGlobal.h"

SVNewtonGlobal::SVNewtonGlobal(const std::function<double(double)>& f, const std::function<double(double)>& f_prime, 
	double tol, size_t max_iter, double starting_point): SVNewton(f, f_prime, tol, 0.05, max_iter, starting_point) {}

void SVNewtonGlobal::setParams(double tol_, size_t max_iter_, double starting_point_)
{
	tol = tol_;
	max_iter = max_iter_;
	starting_point = starting_point_;
	result = 0.0;
	iter_counter = 0;
	multiplicity = -1.0;
	error = 0.0;
}

void SVNewtonGlobal::solve()
{
	iter_counter = 0;

	double x_prev_prev = starting_point;
	double x_prev = x_prev_prev;
	double x_cur = x_prev;

	double f_prev = f(x_prev);
	double f_cur = f_prev;

	double f_prime_prev = f_prime(x_prev);

	error = 1.0;
	double step = 1.0;

	x_cur = x_prev - f_prev / f_prime_prev;
	f_cur = f(x_cur);
	while (abs(f_cur) > abs(f_prev))
	{
		step /= 2.0;
		x_cur = x_prev - step * f_prev / f_prime_prev;
		f_cur = f(x_cur);
	}
	x_prev = x_cur;
	++iter_counter;

	while (error > tol && iter_counter < max_iter)
	{
		step = 1.0;
		f_prev = (iter_counter == 1) ? f(x_prev) : f_cur;
		f_prime_prev = f_prime(x_prev);
		x_cur = x_prev - step * f_prev / f_prime_prev;
		f_cur = f(x_cur);

		while (abs(f_cur) >= abs(f_prev) && step >= 1e-15)
		{
			step /= 2.0;
			x_cur = x_prev - step * f_prev / f_prime_prev;
			f_cur = f(x_cur);
		}

		multiplicity = abs(1.0 / (1.0 - (x_cur - x_prev) / (x_prev - x_prev_prev)));
		error = abs((x_cur - x_prev) * multiplicity);

		x_prev_prev = x_prev;
		x_prev = x_cur;

		++iter_counter;
	}
	result = x_cur;
}
