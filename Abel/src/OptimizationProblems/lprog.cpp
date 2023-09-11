#include "pch.h"
#include "OptimizationProblems/lprog.h"

lprog_Result lprog(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	// Given an LP c*x -> min subject to Ax=b, Bx<=d
	// transform it into standard form and solve, x = x^+ - x^-
	// (c, -c, 0)*(x^+, x^-, y) -> min
	// subject to
	// [[A -A 0]  = (b)
	//  [B -B I]] = (d)
	// x^+, x^-, y >= 0

	int m = (int)A.rows();
	int n = (int)A.cols();
	int p = (int)B.rows();
	int k = (int)B.cols(); // these are assumed to be what determines dimension of the problem

	bool valid_Ab = false;
	bool valid_Bd = ((p == d.rows() && p != 0 && k != 0)) ? true : false; // pair (B,d) is valid if B is non-empty and has as many rows as d
	bool extra_cols = (n < k) ? true : false; // check if matrix A needs to be expanded to meet number of columns of B

	// if A and b are empty or A has as many rows as b and non-empty and has no more rows than B has columns, then it's a valid pair (A,b)
	if ((m == 0 && n == 0 && b.rows() == 0) || (m == b.rows() && m != 0 && n != 0 && m <= k)) valid_Ab = true;

	if (!(valid_Ab && valid_Bd)) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM); // throw if any of the pairs (A,b) or (B,d) are invalid

	Eigen::MatrixXd mat;
	Eigen::VectorXd rhs;
	Eigen::VectorXd objective;

	int dim = 2 * k + p;

	if (m == 0 || n == 0 || b.rows() == 0) { // if (A,b) is empty pair then transform c*x->min, Bx<=d LP to standard form by substitution x = x^+ - x^-
		mat.resize(p, 2 * k + p);
		mat << B, -B, Eigen::MatrixXd::Identity(p, p);
		rhs = d;
		// mat = [B, -B, 0]
	}
	else if (extra_cols) { // if A is non-empty, but has less columns than B, then expand A by adding extra zero columns and transform the problem into standard form
		mat.resize(m + p, 2 * k + p);
		mat << A, Eigen::MatrixXd::Zero(m, k - n), -A, Eigen::MatrixXd::Zero(m, k - n), Eigen::MatrixXd::Zero(m, p),
			   B, -B, Eigen::MatrixXd::Identity(p, p);
		rhs.resize(b.rows() + d.rows());
		rhs << b, d;
		// mat = [[[A, 0], [-A, 0], 0], [B -B I]]
	}
	else { // if A is non-empty and has as many columns as B, then don't add zeros and transform the problem into standard form
		mat.resize(m + p, 2 * k + p);
		mat << A, -A, Eigen::MatrixXd::Zero(m, p), 
			   B, -B, Eigen::MatrixXd::Identity(p, p);
		rhs.resize(b.rows() + d.rows());
		rhs << b, d;
		// mat = [[A, -A, 0], [B, -B, I]]
	}
	objective.resize(2 * c.rows() + p); // new objective vector in standard form is (c, -c, 0)
	objective << c, -c, Eigen::VectorXd::Zero(p, 1);

	MVIntPointLP ip(objective, mat, rhs, dim, tol, max_iter);

	ip.solve();

	Eigen::VectorXd res = ip.getResult();

	// if given a set of r inequalities, then the last r dual variables represent whether the constraint is active (positive) or inactive (close to zero)
	return lprog_Result(res.block(0, 0, k, 1) - res.block(k, 0, k, 1), ip.getDual(), ip.getError(), ip.niter(), ip.isDiverging(), mat.rows(), mat.cols());
}

lprog_Result lprog(const Eigen::VectorXd& c, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, double tol, size_t max_iter)
{
	return lprog(c, Eigen::MatrixXd(), Eigen::VectorXd(), B, d, tol, max_iter);
}