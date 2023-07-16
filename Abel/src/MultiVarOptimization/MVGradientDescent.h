#pragma once

#include "Eigen/Dense"
#include "Base/MVNumericalMethod.h"

class MVGradientDescent : public MVNumericalMethod {

protected:
	std::function<double(const Eigen::VectorXd& x)> f; // primary differentiable objective, could be used as an alternative for adaptive restarts
	std::function<double(const Eigen::VectorXd& x)> g; // secondary convex non-differentiable objective, add me to copy constructor and = operator
	std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad; // gradient of the function f
	std::function<void(Eigen::VectorXd&, double)> prox = [](Eigen::VectorXd& x, double step) {}; // proximal operator for step * g function
	Eigen::VectorXd starting_point;
	double error; // convergence criterion
	double step; // step length
	double gamma = 0.5; // non-convex backtracking parameter
	double eta = 2.0; // backtracking parameter
	double L = 1.0; // Lipschitz constant
	bool isConvex = false; // is the underlying problem convex?
	bool isConstStep = false; // should the step size be constant (1/L)?
	bool hasStartingPoint = false;

public: 
	MVGradientDescent(const std::function<double(const Eigen::VectorXd& x)>& f,
		const std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)>& f_grad, size_t dimension,
		std::function<double(const Eigen::VectorXd& x)> g = [](const Eigen::VectorXd& x) { return 0.0; },
		double tol = 1e-5, double step = 0.001, int max_iter = 100) :
		f(f), g(g), f_grad(f_grad), error(-1.0), step(std::min(std::max(0.0, step) + 0.0001, 1.0)), starting_point(Eigen::VectorXd(dimension)), MVNumericalMethod(dimension, tol, max_iter) {}
	
	MVGradientDescent(const MVGradientDescent& gr) : f(gr.f), g(gr.g), f_grad(gr.f_grad), prox(gr.prox), starting_point(gr.starting_point), error(gr.error),
		step(gr.step), isConvex(gr.isConvex), isConstStep(gr.isConstStep), MVNumericalMethod(gr.dimension, gr.tol, gr.max_iter, gr.was_run, gr.iter_counter, gr.result) {}

	MVGradientDescent& operator=(const MVGradientDescent& gr) {
		f = gr.f;
		g = gr.g;
		f_grad = gr.f_grad;
		prox = gr.prox;
		starting_point = gr.starting_point;
		error = gr.error;
		step = gr.step;
		tol = gr.tol;
		max_iter = gr.max_iter;
		iter_counter = gr.iter_counter;
		dimension = gr.dimension;
		result = gr.result;
		was_run = gr.was_run;
		isConvex = gr.isConvex;
		isConstStep = gr.isConstStep;
		hasStartingPoint = gr.hasStartingPoint;
		return *this;
	}

	void solve() override {

		// FISTA algorithm with adaptive restarts and backtracking of Lipschitz constant L
		// Reference:
		// - Adaptive Restart for Accelerated Gradient Schemes (Brendan O'Donoghue, Emmanuel Candes) - https://arxiv.org/abs/1204.3982
		// - Lipschitz constant backtracking due to "Amir Beck, First-Order Methods in Optimization" Chapter 10, p. [10,15,23] - https://archive.siam.org/books/mo25/

		if (was_run) { return; }
		if (!hasStartingPoint) { starting_point.setRandom(); }

		// variables
		Eigen::VectorXd x = starting_point;
		Eigen::VectorXd x_prev = x;
		Eigen::VectorXd y = x;
		Eigen::VectorXd y_prev = x;

		Eigen::VectorXd grad(dimension); // gradient
		Eigen::VectorXd G(dimension); // generalized gradient (gradient mapping)
		Eigen::VectorXd z(dimension); // temporary variable

		double theta = 1.0; // initial values of theta and beta for accelerated prox scheme of FISTA
		double theta_prev = theta;
		double beta = 1.0;
		double restart_criterion = 0.0; // when this is positive a restart happens

		do
		{
			y_prev = y;
			L = isConstStep ? 1.0 / step : 1.0; // if there's a const step enabled choose L accordingly (step = 1/L), otherwise set L = 1.0
			f_grad(grad, y_prev);
			y = y_prev - grad / L;
			prox(y, 1.0 / L); // evaluate next prox iterate
			if (isConstStep && !isConvex) {
				G = L * (y - y_prev);
			}
			else {
				G = y - y_prev;
			}

			error = G.squaredNorm();
			if (error <= 1e-15) { x = y; break; } // safety
			if (!isConstStep && isConvex)
			{
				while (f(y) > f(y_prev) + grad.dot(G) + L / 2.0 * error) // backtracking for convex case
				{
					L *= eta;
					y = y_prev - grad / L;
					prox(y, 1.0 / L);
					G = (y - y_prev);
					error = G.squaredNorm();
				}
			}
			else if (!isConstStep && !isConvex)
			{
				while (f(y_prev) + g(y_prev) - f(y) - g(y) < gamma / L * error) // backtracking for non-convex case
				{
					L *= eta;
					y = y_prev - grad / L;
					prox(y, 1.0 / L);
					G = L * (y - y_prev);
					error = G.squaredNorm();
				}
			}

			error = sqrt(error);
			x_prev = x;
			x = y;
			theta = (1.0 + sqrt(1.0 + 4.0 * pow(theta_prev, 2.0))) / 2.0;
			beta = (theta_prev - 1.0) / theta;
			theta_prev = theta;
			z = x - x_prev;
			y = x + beta * (z);

			restart_criterion = isConvex ? G.dot(z) * (-L) : -(G.dot(z)); // restart criterion for different cases of generalized gradient

			if (restart_criterion > 0) { // restart whenever this is positive
				x = x_prev;
				y = x;
				theta = 1.0;
				theta_prev = theta;
			}
			// for debugging purposes
			//std::cout << "L: " << L << " Beta: " << beta << " Theta: " << theta << " Error: " << error << " Restart: " << (restart_criterion > 0) << '\n';
			//std::cout << x << '\n';
			++iter_counter;
		} while (error >= tol && iter_counter < max_iter);

		// for debugging purposes
		//std::cout << "This was convex(" << isConvex << "), constant step(" << isConstStep << ')' << '\n';

		result = x;
		was_run = true;
	}
	// prox operator must be passed with type "double" parameter because in proximal gradient descent 
	// if you have a step \alpha and a function g(x) for which you want to compute proximal operator, it's the same as computing proximal gradient for alpha * g
	// (see LaTeX below)
	// the convention is that projection operators (prox of indicator function) still have the alpha, but don't use it
	// while Abel's own prox operators already have necessary slot for such scalar
	// i. e. you can create a lambda function and call L1Prox(x, alpha * beta) inside it for user defined parameter "beta"
	// --- LaTeX formatting ---
	// x_{k+1} = prox_{g}(y_k) = \underset{u \in \mathbb{R}^n}{\arg \min}(g(u) + \frac{1}{2\alpha}||u-y_k||_2^2) = 
	// \underset{u \in \mathbb{R}^n}{\arg \min}(\alpha g(u) + \frac{1}{2}||u-y_k||_2^2) = prox_{\alpha g}(y_k)
	void setProx(const std::function<void(Eigen::VectorXd& x, double alpha)>& Prox) { prox = Prox; }

	void setParams(double tol_ = 1e-5, size_t max_iter_ = 100, double step_ = 0.001) {
		tol = std::max(1e-15, tol_);
		max_iter = std::max(2, max_iter);
		iter_counter = 0;
		was_run = false;
		step = std::min(std::max(0.0, step) + 0.0001, 1.0);
	}
	void setStart(const Eigen::VectorXd& x) {
		if (dimension != x.rows()) throw 1;
		starting_point = x;
		hasStartingPoint = true;
	}
	void toggleConstStep() {
		isConstStep = isConstStep ? false : true;
	}
	void toggleConvex() {
		isConvex = isConvex ? false : true;
	}
	double getError() { return error; }
};