#pragma once

#include "Base/MVNumericalMethod.h"

class MVIntPointLP : public MVNumericalMethod {

protected:
	bool isDivergent = false;

	Eigen::VectorXd dual_variables; // vector of dual variables

	const Eigen::MatrixXd* A; // matrix in the set of linear equalities Ax=b
	const Eigen::VectorXd* b; // vector in right hand side of the set of linear equalities Ax=b
	const Eigen::VectorXd* c; // vector in the objective function

public:

	MVIntPointLP(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
		size_t dimension, double tol = 1e-10, size_t max_iter = 100);

	MVIntPointLP(const MVIntPointLP& ip);

	MVIntPointLP& operator=(const MVIntPointLP& ip);

	void solve() noexcept override;

	void setStart(const Eigen::VectorXd& x, const Eigen::VectorXd& z, const Eigen::VectorXd& s);
	void setParams(double tol_ = 1e-10, size_t max_iter_ = 100) noexcept;
	Eigen::VectorXd& getDual() noexcept;
	bool isDiverging() const noexcept;

protected:
	int smallest_ratio_index(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) const;

private:
	void phase1(Eigen::VectorXd& x, Eigen::VectorXd& z, Eigen::VectorXd& s) noexcept;
	void phase2(Eigen::VectorXd& x, Eigen::VectorXd& z, Eigen::VectorXd& s) noexcept;
};