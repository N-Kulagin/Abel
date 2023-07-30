#include "pch.h"
#include "MultiVarOptimization/MVNewton.h"

MVNewton::MVNewton(
	std::function<double(const Eigen::VectorXd& x)> f, 
	std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad,
	std::function<void(Eigen::MatrixXd& H, const Eigen::VectorXd& input)> f_hess, 
	size_t dimension, double tol, int max_iter) : f(f), f_grad(f_grad), f_hess(f_hess), 
	Hessian(Eigen::MatrixXd(dimension,dimension)), grad(Eigen::VectorXd(dimension)),
	MVNumericalMethod(dimension, tol, max_iter) {}

MVNewton::MVNewton(const MVNewton& n) : alpha(n.alpha), beta(n.beta), hasConstraints(n.hasConstraints),
	f(n.f), f_grad(n.f_grad), f_hess(n.f_hess), Hessian(n.Hessian), grad(n.grad), dual_variables(n.dual_variables), 
	A(n.A), A_t(n.A_t), b(n.b), 
	MVNumericalMethod(n.dimension,n.tol,n.max_iter,n.was_run,n.iter_counter,n.error,n.result,n.starting_point,n.hasStart) {}

MVNewton& MVNewton::operator=(const MVNewton& n)
{
	MVNumericalMethod::operator=(n);
	alpha = n.alpha;
	beta = n.beta;
	hasConstraints = n.hasConstraints;
	f = n.f;
	f_grad = n.f_grad;
	f_hess = n.f_hess;
	Hessian = n.Hessian;
	grad = n.grad;
	dual_variables = n.dual_variables;
	A = n.A;
	A_t = n.A_t;
	b = n.b;

	return *this;
}

void MVNewton::setConstraints(const Eigen::MatrixXd& A_mat, const Eigen::VectorXd& b_vec)
{
	if (A_mat.cols() != dimension || b_vec.rows() != A_mat.rows() || A_mat.rows() > A_mat.cols()) throw 1;
	hasConstraints = true;
	was_run = false;
	A = A_mat;
	A_t = A.transpose();
	b = b_vec;
}

void MVNewton::setStart(const Eigen::VectorXd& x)
{
	if (x.size() != dimension) throw 1;
	hasStart = true;
	was_run = false;
	starting_point = x;
}

void MVNewton::solve() noexcept
{
	if (was_run) return;
	iter_counter = 0;
	error = 1.0;
	if (hasConstraints) solve_Constrained();
	else solve_Unconstrained();
	was_run = true;
}

Eigen::VectorXd& MVNewton::getDual()
{
	return dual_variables;
}

void MVNewton::solve_Constrained() noexcept
{
	int A_rows = (int)A.rows();

	Eigen::VectorXd x(dimension);
	if (hasStart) x = starting_point;
	else x.setRandom();

	Eigen::VectorXd mu(A_rows);
	Eigen::VectorXd residual(dimension + A_rows);
	Eigen::VectorXd solution(dimension + A_rows);
	Eigen::MatrixXd KKT_Matrix(dimension + A_rows, dimension + A_rows);
	Eigen::MatrixXd zero(A_rows, A_rows);

	mu.setRandom();
	Eigen::VectorXd x_prev = x;
	Eigen::VectorXd mu_prev = mu;
	solution.setRandom();
	zero.setZero();

	KKT_Matrix.block(0, dimension, dimension, A_rows) = A_t;
	KKT_Matrix.block(dimension, 0, A_rows, dimension) = A;
	KKT_Matrix.block(dimension, dimension, A_rows, A_rows) = zero;

	double step = 1.0;

	do {
		if (iter_counter == 0) {
			f_grad(grad, x);

			residual.block(0, 0, dimension, 1) = -(grad + A_t * mu);
			residual.block(dimension, 0, A_rows, 1) = -(A * x - b);
		}
		f_hess(Hessian, x);

		KKT_Matrix.block(0, 0, dimension, dimension) = Hessian;

		solution = KKT_Matrix.partialPivLu().solve(residual);

		step = 1.0 / beta;
		x_prev = x;
		mu_prev = mu;
		error = residual.norm();
		do
		{
			step *= beta;

			x = x_prev + step * solution.block(0, 0, dimension, 1);
			mu = mu_prev + step * solution.block(dimension, 0, A_rows, 1);
			f_grad(grad, x);
			residual.block(0, 0, dimension, 1) = -(grad + A_t * mu);
			residual.block(dimension, 0, A_rows, 1) = -(A * x - b);

		} while (residual.norm() > (1.0 - alpha * step) * error);

		++iter_counter;
	} while (error / 2.0 >= tol && iter_counter <= max_iter);
	
	error = residual.norm();
	result = x;
	dual_variables = mu;
}

void MVNewton::solve_Unconstrained() noexcept
{
	Eigen::VectorXd x(dimension);
	if (hasStart) x = starting_point;
	else x.setRandom();

	Eigen::VectorXd x_prev(dimension);
	Eigen::VectorXd solution(dimension);

	double step = 1.0;

	while (error/2.0 >= tol && iter_counter < max_iter)
	{
		f_grad(grad, x);
		f_hess(Hessian, x);

		solution = Hessian.partialPivLu().solve(-grad);

		step = 1.0;
		x_prev = x;
		x += step * solution;

		while (f(x) > f(x_prev) + alpha * step * grad.dot(solution))
		{
			step *= beta;
			x = x_prev + step * solution;
		}
		error = -(grad.dot(solution));
		++iter_counter;
	}
	result = x;
}

void MVNewton::setParams(double tol_, size_t max_iter_, double alpha_, double beta_)
{
	tol = std::max(1e-15, tol_);
	max_iter = std::max(2, max_iter);
	alpha = std::min(std::max(1e-10,alpha_),0.499);
	beta = std::min(std::max(1e-3,beta_),0.999);
	iter_counter = 0;
	was_run = false;
}