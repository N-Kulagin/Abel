#include "pch.h"
#include "RootFinding/SVNewton.h"

SVNewton::SVNewton(const std::function<double(double)>& f, const std::function<double(double)>& f_prime, double tol, 
	double heuristic, size_t max_iter, double starting_point): 
	f(f), f_prime(f_prime), heuristic(heuristic), starting_point(starting_point), multiplicity(-1.0), SVNumericalMethod(tol, max_iter) {}

SVNewton::SVNewton(const SVNewton& n): f(n.f), f_prime(n.f_prime), heuristic(n.heuristic), starting_point(n.starting_point),
multiplicity(n.multiplicity), SVNumericalMethod(n.tol, n.max_iter, n.was_run, n.iter_counter, n.result, n.error) {}

SVNewton& SVNewton::operator=(const SVNewton& n)
{
	f = n.f;
	f_prime = n.f_prime;
	heuristic = n.heuristic;
	starting_point = n.starting_point;
	multiplicity = n.multiplicity;
	SVNumericalMethod::operator=(n);

	return *this;
}

void SVNewton::setParams(double tol_, double heuristic_, size_t max_iter_, double starting_point_)
{
	tol = tol_;
	heuristic = heuristic_;
	max_iter = max_iter_;
	starting_point = starting_point_;
	was_run = false;
	result = 0.0;
	iter_counter = 0;
	multiplicity = -1.0;
	error = 0.0;
}

double SVNewton::getMultiplicity()
{
	return multiplicity;
}

void SVNewton::solve()
{
	if (was_run) return;
	iter_counter = 0;

	double x_prev_prev = starting_point;
	double x_prev = x_prev_prev;
	double x_cur = x_prev;

	double f_prev = f(x_prev);
	double f_cur = f(x_cur);

	double f_prime_prev = f_prime(x_prev);

	double psi_0 = 0.0;
	double psi_1 = 0.0;

	double tau = 1.0;

	error = 1.0;

	// one iteration
	x_cur = x_prev - f_prev / f_prime_prev;
	f_cur = f(x_cur);
	psi_0 = pow(f_prev, 2);
	psi_1 = pow(f_cur, 2);
	tau = (psi_0 + heuristic * psi_1) / (psi_0 + psi_1);
	x_cur = x_prev - tau * f_prev / f_prime_prev;
	x_prev = x_cur;
	++iter_counter;
	//

	while (error > tol && iter_counter < max_iter)
	{
		f_prev = f(x_prev);
		f_prime_prev = f_prime(x_prev);
		x_cur = x_prev - f_prev / f_prime_prev;
		f_cur = f(x_cur);

		psi_0 = pow(f_prev, 2);
		psi_1 = pow(f_cur, 2);
		tau = (psi_0 + heuristic * psi_1) / (psi_0 + psi_1);
		x_cur = x_prev - tau * f_prev / f_prime_prev;

		multiplicity = abs(1.0 / (1.0 - (x_cur - x_prev) / (x_prev - x_prev_prev)));
		error = abs((x_cur - x_prev) * multiplicity);

		x_prev_prev = x_prev;
		x_prev = x_cur;

		++iter_counter;
	}
	result = x_cur;
	was_run = 1;
}

std::ostream& operator<<(std::ostream& out, const SVNewton& n) {
	out << "Newton information:" << '\n'
		<< "Tolerance: " << n.tol << '\n'
		<< "Heuristic: " << n.heuristic << '\n'
		<< "Max iterations: " << n.max_iter << '\n'
		<< "Performed iterations: " << n.iter_counter << '\n'
		<< "Starting point: " << n.starting_point;
	return out;
}


