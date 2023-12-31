#pragma once

class MVNelderMead : public MVNumericalMethod {

private:
	std::function<double(const Eigen::VectorXd& x)> f;
	double alpha = 1.0; // reflection coefficient, alpha > 0
	double beta = 0.5; // contraction coefficient, 0 < beta < 1
	double gamma = 2.0; // stretching/expansion coefficient, gamma > 1 and gamma > alpha
	double delta = 0.5; // shrinkage coefficient, 0 < delta < 1
	double randomCoeff = 1.0; // determines the spread of randomized initial simplex
	Eigen::MatrixXd* x = nullptr; // simplex


public:
	MVNelderMead(std::function<double(const Eigen::VectorXd& x)> f, size_t dimension, double tol = 1e-3, int max_iter = 100, bool hasLog = false);

	MVNelderMead(const MVNelderMead& nm);

	MVNelderMead operator=(const MVNelderMead& nm);

	void solve() noexcept override;

	void setParams(double tol = 1e-3, int max_iter = 100, double randomCoeff_ = 1.0, double alpha_ = 1.0, double beta_ = 0.5, double gamma_ = 2.0, double delta_ = 0.5);

	void setStart(Eigen::MatrixXd& simplex);

	void printLogs() const override;

private:
	void solve_(Eigen::MatrixXd& x);
	int findSecondWorst(const Eigen::ArrayXd& f_vals, double abs_max, int dimension);
	void findMaxMin(const Eigen::ArrayXd& f_vals, double& abs_max, double& abs_min, int& worstVertexIndex, int& bestVertexIndex, int dimension);
	void shrinkAndRecalculate(Eigen::MatrixXd& x, Eigen::ArrayXd& f_vals, double& abs_max, double& abs_min, int& worstVertexIndex, int& bestVertexIndex, int dimension);
};