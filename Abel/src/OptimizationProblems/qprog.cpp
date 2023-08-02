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

	if (G.rows() != k || G.cols() != k) throw 1;

	bool valid_Ab = (m == b.rows() && m != 0 && n != 0 && m <= k) ? true : false;
	bool valid_Bd = (p == d.rows() && p != 0 && k != 0) ? true : false;
	bool valid_c = (c_rows == 0 || c_rows == k) ? true : false;

	if (!(valid_Ab && valid_Bd && valid_c)) throw 1;

	Eigen::VectorXd cons;
	if (c_rows == 0) {
		cons.resize(k);
		cons.setZero();
	}

	MVIntPointQP ip = (c_rows != 0) ? MVIntPointQP(G, c, A, b, B, d, k, tol, max_iter) : MVIntPointQP(G, cons, A, b, B, d, k, tol, max_iter);

	ip.solve();

	return qprog_Result(ip.getResult(),ip.getDual(),ip.getError(),ip.niter(),ip.isDiverging(),m,p);
}

qprog_Result qprog(const Eigen::MatrixXd& G, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	return qprog(G, Eigen::VectorXd(), A, b, B, d, tol, max_iter);
}
