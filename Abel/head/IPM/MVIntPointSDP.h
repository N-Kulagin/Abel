#pragma once
#include "Base/MVNumericalMethod.h"

class MVIntPointSDP : public MVNumericalMethod {

public:
	Eigen::MatrixXd primal_variable_X; // primal positive-semidefinite matrix in (X,y,S) triplet
	Eigen::MatrixXd dual_variable_S; // dual positive-semidefinite matrix in (X,y,S) triplet
	Eigen::VectorXd dual_variable_y; // dual vector of lagrange multipliers in (X,y,S) triplet

private:
	bool isDivergent = false; // divergence flag
	std::vector<Eigen::MatrixXd*> v; // vector that constains references to objective matrix C and matrices in constraints A_i, C is assumed to be first reference
	const Eigen::VectorXd* b = nullptr; // right hand side of constraints <A_i, X> = b_i
	Eigen::MatrixXd X_start; // starting matrix X

public:
	MVIntPointSDP(const std::vector<Eigen::MatrixXd*>& v, const Eigen::VectorXd& b, size_t dimension, double tol = 1e-10, size_t max_iter = 100, bool hasLog = false);
	MVIntPointSDP(const MVIntPointSDP& s);
	MVIntPointSDP& operator=(const MVIntPointSDP& sdp);
	void solve() noexcept override;
	void setParams(double tol_, size_t max_iter_) noexcept;
	void setStart(const Eigen::MatrixXd& X_);
	bool isDiverging() const noexcept;
	void printLogs() const override;
};