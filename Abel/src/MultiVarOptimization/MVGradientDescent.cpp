#include "pch.h"
#include "MultiVarOptimization/MVGradientDescent.h"

MVGradientDescent::MVGradientDescent(
	const std::function<double(const Eigen::VectorXd& x)>& f, const std::function<void(Eigen::VectorXd& grad,
	const Eigen::VectorXd& input)>& f_grad, size_t dimension, std::function<double(const Eigen::VectorXd& x)> g, 
	double tol, double step, int max_iter, bool hasLog) :
	f(f), g(g), f_grad(f_grad), prox([](Eigen::VectorXd& x, double step) {}), step(std::min(std::max(0.0, step) + 1e-10, 1.0)),
	MVNumericalMethod(dimension, tol, max_iter, hasLog) {
	if (hasLog) {
		lg = AbelLogger(4);
	}
}

MVGradientDescent::MVGradientDescent(const MVGradientDescent& gr) : f(gr.f), g(gr.g), f_grad(gr.f_grad), prox(gr.prox), 
step(gr.step), isConvex(gr.isConvex), isConstStep(gr.isConstStep),
MVNumericalMethod(gr.dimension, gr.tol, gr.max_iter, gr.hasLog, gr.iter_counter, gr.error, gr.result, gr.starting_point, gr.hasStart, gr.lg) {}

MVGradientDescent& MVGradientDescent::operator=(const MVGradientDescent& gr)
{
	MVNumericalMethod::operator=(gr);
	f = gr.f;
	g = gr.g;
	f_grad = gr.f_grad;
	prox = gr.prox;
	step = gr.step;
	isConvex = gr.isConvex;
	isConstStep = gr.isConstStep;
	return *this;
}

void MVGradientDescent::solve() noexcept
{
	// FISTA algorithm with adaptive restarts and backtracking of Lipschitz constant L
	// Reference:
	// - Adaptive Restart for Accelerated Gradient Schemes (Brendan O'Donoghue, Emmanuel Candes) - https://arxiv.org/abs/1204.3982
	// - Lipschitz constant backtracking due to "Amir Beck, First-Order Methods in Optimization" Chapter 10, p. [10,15,23] - https://archive.siam.org/books/mo25/

	if (!hasStart) { starting_point = Eigen::VectorXd(dimension); starting_point.setRandom(); }

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
		L = isConstStep ? 1.0 / step : L; // if there's a const step enabled choose L accordingly (step = 1/L), otherwise set L = 1.0 (default) or equal to previous L
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
		if (error <= 1e-15) { ++iter_counter; x = y; break; } // safety
		if (!isConstStep && isConvex)
		{
			while (f(y) > f(y_prev) + grad.dot(G) + L / 2.0 * error && L < 1e+10) // backtracking for convex case
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
			while (f(y_prev) + g(y_prev) - f(y) - g(y) < gamma / L * error && L < 1e+10) // backtracking for non-convex case
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
		//theta = (1.0 + sqrt(1.0 + 4.0 * pow(theta_prev, 2.0))) / 2.0; <--- this update scheme is used in the referenced paper
		theta = theta_prev + 0.5; // <--- asymptotically the same as above, but uses simpler update scheme
		beta = (theta_prev - 1.0) / theta;
		theta_prev = theta;
		z = x - x_prev;
		y = x + beta * (z);

		restart_criterion = isConvex ? G.dot(z) * (-L) : -(G.dot(z)); // restart criterion for different cases of generalized gradient

		// after L has been increased by backtracking, decrease it a bit to allow for both decreasing and increasing step lengths
		// Optimization and Statistics - Лекции по курсу численные методы оптимизации ФУПМ 2022. Лекция 5
		// https://www.youtube.com/live/JOLoR5Io4AQ?si=7MLeJOEHXR7YirNJ&t=3293
		L /= eta;

		if (hasLog) {
			lg.record(f(y)+g(y), 0);
			lg.record(error, 1);
			lg.record(1.0 / L, 2);
			lg.record(restart_criterion, 3);
		}

		if (restart_criterion > 0) { // restart whenever this is positive
			x = x_prev;
			y = x;
			theta = 1.0;
			theta_prev = theta;
		}
		++iter_counter;
	} while (error >= tol && iter_counter < max_iter);

	result = x;
	hasStart = false;
}

void MVGradientDescent::setProx(const std::function<void(Eigen::VectorXd& x, double alpha)>& Prox)
{
	prox = Prox;
}

void MVGradientDescent::setParams(double tol_, size_t max_iter_, double step_) noexcept
{
	tol = std::max(1e-15, tol_);
	max_iter = std::max((size_t)2, max_iter);
	iter_counter = 0;
	step = std::min(std::max(0.0, step) + 0.0001, 1.0);
}

void MVGradientDescent::setStart(const Eigen::VectorXd& x)
{
	if (dimension != x.rows()) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	starting_point = x;
	hasStart = true;
}

void MVGradientDescent::toggleConstStep() noexcept
{
	isConstStep = isConstStep ? false : true;
}

void MVGradientDescent::toggleConvex() noexcept
{
	isConvex = isConvex ? false : true;
}

void MVGradientDescent::printLogs() const
{
	lg.print("MVGradientDescent", { "ObjectiveValue", "GradientNorm", "Step", "RestartCriterion" });
}

