#pragma once

#include "MVIntPointLP.h"

class MVIntPointQP : public MVIntPointLP {

private:
	const Eigen::MatrixXd* G;
	const Eigen::MatrixXd* B;
	Eigen::MatrixXd B_t;
	const Eigen::VectorXd* d;

public:
	MVIntPointQP(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& A, 
		const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d,
		size_t dimension, double tol = 1e-10, size_t max_iter = 100);

	MVIntPointQP(const MVIntPointQP& ip);

	MVIntPointQP& operator=(const MVIntPointQP& ip);

	void solve() noexcept override;
	void setStart(const Eigen::VectorXd& x, const Eigen::VectorXd& l, const Eigen::VectorXd& z, const Eigen::VectorXd& y);
};