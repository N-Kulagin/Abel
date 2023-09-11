#include "pch.h"
#include "OptimizationProblems/qprog.h"

qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, 
	double tol, size_t max_iter)
{

	int m = (int)A.rows();
	int n = (int)A.cols();
	int p = (int)B.rows();
	int k = (int)B.cols();
	int c_rows = (int)c.rows();

	if (G.rows() != k || G.cols() != k) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);

	// pair (A,b) is valid if A has as many rows as b, A has no more rows than B has columns (columns of B are assumed to be what determines dimension of the problem)
	// pair (B,d) is considered valid if B has as many rows as d and B is not an empty matrix
	// vector c is considered valid if it is either empty or has as many rows as B has columns

	bool valid_Ab = (m == b.rows() && m <= k) ? true : false;
	bool valid_Bd = (p == d.rows() && p != 0 && k != 0) ? true : false;
	bool valid_c = (c_rows == 0 || c_rows == k) ? true : false;

	if (!(valid_Ab && valid_Bd && valid_c)) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);

	Eigen::VectorXd cons; // if c is empty vector, use zero vector instead
	if (c_rows == 0) {
		cons.resize(k);
		cons.setZero();
	}
	// depending on what problem was given (with or without equality constraints, with or without c vector), setup appropriate interior point method
	MVIntPointQP ip = (c_rows != 0 && A.size() != 0) ? MVIntPointQP(G, c, A, b, B, d, k, tol, max_iter) :
		(c_rows != 0 && A.size() == 0) ? MVIntPointQP(G, c, B, d, k, tol, max_iter) :
		(c_rows == 0 && A.size() != 0) ? MVIntPointQP(G, cons, A, b, B, d, k, tol, max_iter) :
										 MVIntPointQP(G, cons, B, d, k, tol, max_iter);

	ip.solve();

	return qprog_Result(ip.getResult(),ip.getDual(),ip.getError(),ip.niter(),ip.isDiverging(),m,p);
}


qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	return qprog(G, Eigen::VectorXd(), A, b, B, d, tol, max_iter);
}

qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	return qprog(G, Eigen::VectorXd(), Eigen::MatrixXd(), Eigen::VectorXd(), B, d, tol, max_iter);
}

qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	return qprog(G, c, Eigen::MatrixXd(), Eigen::VectorXd(), B, d, tol, max_iter);
}
