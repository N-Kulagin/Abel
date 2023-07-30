#pragma once

#include "Base/MVNumericalMethod.h"

class MVGradientDescent : public MVNumericalMethod {

protected:
	std::function<double(const Eigen::VectorXd& x)> f; // primary differentiable objective, could be used as an alternative for adaptive restarts
	std::function<double(const Eigen::VectorXd& x)> g; // secondary convex non-differentiable objective, add me to copy constructor and = operator
	std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad; // gradient of the function f
	std::function<void(Eigen::VectorXd&, double)> prox; // proximal operator for step * g function
	double step; // step length
	double gamma = 0.5; // non-convex backtracking parameter
	double eta = 2.0; // backtracking parameter
	double L = 1.0; // Lipschitz constant
	bool isConvex = false; // is the underlying problem convex?
	bool isConstStep = false; // should the step size be constant (1/L)?

public:
	MVGradientDescent(const std::function<double(const Eigen::VectorXd& x)>& f,
		const std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)>& f_grad, size_t dimension,
		std::function<double(const Eigen::VectorXd& x)> g = [](const Eigen::VectorXd& x) { return 0.0; },
		double tol = 1e-5, double step = 0.001, int max_iter = 100);

	MVGradientDescent(const MVGradientDescent& gr);

	MVGradientDescent& operator=(const MVGradientDescent& gr);

	void solve() override;

	// prox operator must be passed with type "double" parameter because in proximal gradient descent 
	// if you have a step \alpha and a function g(x) for which you want to compute proximal operator, it's the same as computing proximal gradient for alpha * g
	// (see LaTeX below)
	// the convention is that projection operators (prox of indicator function) still have the alpha, but don't use it
	// while Abel's own prox operators already have necessary slot for such scalar
	// i. e. you can create a lambda function and call L1Prox(x, alpha * beta) inside it for user defined parameter "beta"
	// --- LaTeX formatting ---
	// x_{k+1} = prox_{g}(y_k) = \underset{u \in \mathbb{R}^n}{\arg \min}(g(u) + \frac{1}{2\alpha}||u-y_k||_2^2) = 
	// \underset{u \in \mathbb{R}^n}{\arg \min}(\alpha g(u) + \frac{1}{2}||u-y_k||_2^2) = prox_{\alpha g}(y_k)
	void setProx(const std::function<void(Eigen::VectorXd& x, double alpha)>& Prox);

	void setParams(double tol_ = 1e-5, size_t max_iter_ = 100, double step_ = 0.001);

	void setStart(const Eigen::VectorXd& x);

	void toggleConstStep();

	void toggleConvex();
};