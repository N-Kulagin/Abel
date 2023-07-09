#pragma once

#include "Eigen/Dense"
#include "Base/MVNumericalMethod.h"

class MVGradientDescent : public MVNumericalMethod {

protected:
	std::function<double(const Eigen::VectorXd& x)> f; // could be used as an alternative for adaptive restarts
	std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad;
	std::function<void(Eigen::VectorXd& x)> proj = [](Eigen::VectorXd& x) {};
	Eigen::VectorXd starting_point;
	double error;
	double step;
	bool isProjected = false;

public:
	MVGradientDescent(const std::function<double(const Eigen::VectorXd& x)>& f, const std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)>& f_grad,
		size_t dimension, double tol = 0.01, double step = 0.01, int max_iter = 100, const Eigen::VectorXd& start = Eigen::VectorXd()) :
		f(f), f_grad(f_grad), error(-1.0), starting_point(start), step(std::min(std::max(0.0, step) + 0.0001, 1.0)), MVNumericalMethod(dimension, tol, max_iter) {
		if (starting_point.size() == 0) {
			starting_point.resize(dimension);
			starting_point.setConstant(1.0);
		}
	}
	MVGradientDescent(const MVGradientDescent& g) : f(g.f), f_grad(g.f_grad), proj(g.proj), error(g.error), starting_point(g.starting_point),
		step(g.step), isProjected(g.isProjected), MVNumericalMethod(g.dimension, g.tol, g.max_iter, g.wasRun, g.iter_counter, g.result) {}

	MVGradientDescent& operator=(const MVGradientDescent& g) {
		f = g.f;
		f_grad = g.f_grad;
		proj = g.proj;
		starting_point = g.starting_point;
		step = g.step;
		error = g.error;
		isProjected = g.isProjected;
		tol = g.tol;
		max_iter = g.max_iter;
		iter_counter = g.iter_counter;
		dimension = g.dimension;
		result = g.result;
		wasRun = g.wasRun;
		return *this;
	}

	virtual void solve() override {
		Eigen::VectorXd y = starting_point;
		Eigen::VectorXd x_prev = y;
		Eigen::VectorXd x = y;
		Eigen::VectorXd grad(dimension);
		Eigen::VectorXd z(dimension);

		double theta = 1.0;
		double theta_prev = theta;
		double beta = 1.0;
		double q = isProjected ? 0.0 : 0.001; // 0.001
		error = tol + 0.1;
		iter_counter = 0;

		// Adaptive Restart for Accelerated Gradient Schemes Brendan O'Donoghue, Emmanuel Candes
		// https://arxiv.org/abs/1204.3982v1

		while (iter_counter < max_iter && error >= tol) {

			do
			{
				theta = 0.5 * (sqrt(pow(theta_prev, 4.0) - 2 * q * pow(theta_prev, 2.0) + 4.0 * pow(theta_prev, 2.0) + pow(q, 2.0)) + q - pow(theta_prev, 2.0));
				beta = theta_prev * (1.0 - theta_prev) / (pow(theta_prev, 2.0) + theta);
				theta_prev = theta;

				x_prev = x;
				f_grad(grad, y);
				x = y - step * grad;

				if (isProjected) {
					proj(x);
					z = x - x_prev;
					error = z.norm();
				}
				else {
					z = x - x_prev;
					error = grad.norm();
				}

				y = x + beta * z;
				++iter_counter;
			} while (grad.dot(z) <= 0.0 && error >= tol && iter_counter < max_iter); // grad.dot(z) <= 0.0 && error >= tol
			x_prev = x;
			y = x;
			theta = 1.0;
			theta_prev = theta;
		}
		result = x;
		wasRun = true;
	}

	void setProjection(const std::function<void(Eigen::VectorXd& x)>& Proj) { proj = Proj; isProjected = true; }
	void disableProjection() { isProjected = false; }

	void setParams(double tol_ = 0.01, size_t max_iter_ = 100, double step_ = 0.01, const Eigen::VectorXd& start = Eigen::VectorXd()) {
		tol = std::max(1e-15, tol_);
		max_iter = std::max(2, max_iter);
		iter_counter = 0;
		wasRun = 0;
		step = std::min(std::max(0.0, step) + 0.0001, 1.0);
		if (start.size() == starting_point.size()) {
			starting_point = start;
		}
	}
	double getError() { return error; }
};