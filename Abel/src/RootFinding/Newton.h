#pragma once

#include <iostream>
#include <functional>
#include "../Base/SVNumericalMethod.h"

class SVNewton : public SVNumericalMethod {

protected:
	std::function<double(double)> f;
	std::function<double(double)> f_prime;
	double heuristic;
	double starting_point;
	double multiplicity;
	double error;

public:

	SVNewton(const std::function<double(double)>& f, const std::function<double(double)>& f_prime,
		double tol = 0.01, double heuristic = 0.05, size_t max_iter = 100, double starting_point = 1.0) :
		f(f), f_prime(f_prime), heuristic(heuristic), starting_point(starting_point), multiplicity(-1.0), error(1.0), SVNumericalMethod(tol, max_iter) {}
	
	SVNewton(const SVNewton& n) : f(n.f), f_prime(n.f_prime), heuristic(n.heuristic), starting_point(n.starting_point), multiplicity(n.multiplicity), error(n.error),
		SVNumericalMethod(n.tol, n.max_iter, n.wasRun, n.iter_counter, n.result) {}
	
	
	SVNewton& operator=(const SVNewton& n) {
		f = n.f;
		f_prime = n.f_prime;
		tol = n.tol;
		heuristic = n.heuristic;
		max_iter = n.max_iter;
		iter_counter = n.iter_counter;
		starting_point = n.starting_point;
		wasRun = n.wasRun;
		result = n.result;
		multiplicity = n.multiplicity;
		error = n.error;
	
		return *this;
	}
	
	friend std::ostream& operator<<(std::ostream&, const SVNewton&);
	
	virtual void setParams(double tol_ = 0.01, double heuristic_ = 0.05, size_t max_iter_ = 100, double starting_point_ = 1.0) {
		tol = tol_;
		heuristic = heuristic_;
		max_iter = max_iter_;
		starting_point = starting_point_;
		wasRun = false;
		result = 0.0;
		iter_counter = 0;
		multiplicity = -1.0;
		error = 0.0;
	}

	double getMultiplicity() { return multiplicity; }
	double getError() { return error; }
	
	virtual void solve() {
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
		wasRun = 1;
	}
};

std::ostream& operator<<(std::ostream& out, const SVNewton& n) {
	out << "Newton information:" << '\n'
		<< "Tolerance: " << n.tol << '\n'
		<< "Heuristic: " << n.heuristic << '\n'
		<< "Max iterations: " << n.max_iter << '\n'
		<< "Performed iterations: " << n.iter_counter << '\n'
		<< "Starting point: " << n.starting_point;
	return out;
}

class SVNewtonGlobal : public SVNewton {

public:

	SVNewtonGlobal(const std::function<double(double)>& f, const std::function<double(double)>& f_prime,
		double tol = 0.01, size_t max_iter = 100, double starting_point = 1.0) : SVNewton(f, f_prime, tol, 0.05, max_iter, starting_point) {}

	void setParams(double tol_ = 0.01, size_t max_iter_ = 100, double starting_point_ = 1.0) {
		tol = tol_;
		max_iter = max_iter_;
		starting_point = starting_point_;
		wasRun = false;
		result = 0.0;
		iter_counter = 0;
		multiplicity = -1.0;
		error = 0.0;
	}

	void solve() override {
		iter_counter = 0;

		double x_prev_prev = starting_point;
		double x_prev = x_prev_prev;
		double x_cur = x_prev;

		double f_prev = f(x_prev);
		double f_cur = f(x_cur);

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
			f_prev = f(x_prev);
			f_prime_prev = f_prime(x_prev);
			x_cur = x_prev - step * f_prev / f_prime_prev;
			f_cur = f(x_cur);

			while (abs(f_cur) >= abs(f_prev))
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
		wasRun = 1;
	}
};