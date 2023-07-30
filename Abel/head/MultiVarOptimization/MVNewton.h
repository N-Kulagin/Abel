#pragma once

#include "Base/MVNumericalMethod.h"

class MVNewton : public MVNumericalMethod {

private:
	double alpha = 0.1; // backtracking progress parameter in (0;1/2)
	double beta = 0.5; // backtracking step scaling factor in (0;1)
	bool hasConstraints = false;

	std::function<double(const Eigen::VectorXd& x)> f;
	std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad;
	std::function<void(Eigen::MatrixXd& H, const Eigen::VectorXd& input)> f_hess;

	Eigen::MatrixXd Hessian; // hessian matrix
	Eigen::VectorXd grad; // gradient of the function
	Eigen::VectorXd dual_variables; // vector of dual variables

	Eigen::MatrixXd A; // matrix in the set of linear inequalities Ax=b
	Eigen::MatrixXd A_t; // transposed
	Eigen::VectorXd b; // vector in right hand side of the set of linear inequalities Ax=b

public:
	MVNewton(std::function<double(const Eigen::VectorXd& x)> f, 
		std::function<void(Eigen::VectorXd& grad, const Eigen::VectorXd& input)> f_grad,
		std::function<void(Eigen::MatrixXd& H, const Eigen::VectorXd& input)> f_hess, 
		size_t dimension, double tol = 1e-10, int max_iter = 100);

	MVNewton(const MVNewton& n);

	MVNewton& operator=(const MVNewton& n);

	void setConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	void setStart(const Eigen::VectorXd& x);
	void setParams(double tol = 1e-10, size_t max_iter = 100, double alpha = 0.1, double beta = 0.5);
	void solve() noexcept override;
	Eigen::VectorXd& getDual();

private:
	void solve_Constrained() noexcept;
	void solve_Unconstrained() noexcept;

};