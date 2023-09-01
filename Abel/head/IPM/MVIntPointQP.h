#pragma once

#include "MVIntPointLP.h"
#include "Miscellaneous/AbelLogger.h"

class MVIntPointQP : public MVIntPointLP {

private:
	const Eigen::MatrixXd* G; // matrix in the quadratic objective 1/2 x^T * G x
	const Eigen::MatrixXd* B; // matrix in inequality constraints Bx <= d
	const Eigen::VectorXd* d; // right hand side of inequality ocnstraints

public:
	MVIntPointQP(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& A, 
		const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d,
		size_t dimension, double tol = 1e-10, size_t max_iter = 100, bool hasLog = false);

	MVIntPointQP(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& B, const Eigen::VectorXd& d,
		size_t dimension, double tol = 1e-10, size_t max_iter = 100, bool hasLog = false);

	MVIntPointQP(const MVIntPointQP& ip);

	MVIntPointQP& operator=(const MVIntPointQP& ip);

	void solve() noexcept override;
	void setStart(const Eigen::VectorXd& x, const Eigen::VectorXd& l, const Eigen::VectorXd& z, const Eigen::VectorXd& y);
	void printLogs() const override;
};