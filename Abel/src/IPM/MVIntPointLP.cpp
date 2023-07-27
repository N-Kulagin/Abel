#include "pch.h"
#include "IPM/MVIntPointLP.h"


MVIntPointLP::MVIntPointLP(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, 
	size_t dimension, double tol, int max_iter) : A(A), A_t(A.transpose()), b(b), c(c), MVNumericalMethod(dimension, tol, max_iter) {}

MVIntPointLP::MVIntPointLP(const MVIntPointLP& ip) : A(ip.A), A_t(ip.A_t), b(ip.b), c(ip.c), starting_point(ip.starting_point), hasStart(ip.hasStart), isDivergent(ip.isDivergent),
dual_variables(ip.dual_variables), MVNumericalMethod(ip.dimension,ip.tol,ip.max_iter,ip.was_run,ip.iter_counter,ip.error,ip.result) {}

MVIntPointLP& MVIntPointLP::operator=(const MVIntPointLP& ip)
{
	MVNumericalMethod::operator=(ip);

	A = ip.A;
	A_t = ip.A_t;
	b = ip.b;
	c = ip.c;
	starting_point = ip.starting_point;
	hasStart = ip.hasStart;
	isDivergent = ip.isDivergent;
	dual_variables = ip.dual_variables;

	return *this;
}

void MVIntPointLP::solve()
{
	// Mehrotra's Predictor-Corrector Algorithm for Linear Programming
	// Jorge Nocedal, Stephen Wright - Numerical Optimization, page 411
	// Input problem:
	// c*x -> minimize, subject to
	// Ax = b, x >= 0
	// used convention of a primal-dual pair is (x,z,s) 
	// where x is primal variable, z is dual variable related to equality constraints and s is dual variable related to inequality constraints 

	if (was_run) return;
	if (dimension != A.cols() || dimension < (size_t)A.rows()) throw 1;
	Eigen::VectorXd x(A.cols());
	Eigen::VectorXd z(A.rows());
	Eigen::VectorXd s(A.cols());
	if (!hasStart) phase1(x, z, s); // phase 1 of the algorithm to locate starting point
	else {
		x = starting_point.block(0, 0, (int)dimension, 1); // use user-supplied starting point (x,z,s) (primal, dual equality, dual inequality)
		z = starting_point.block(dimension, 0, A.rows(), 1);
		s = starting_point.block(dimension + A.rows(), 0, dimension, 1);
	}
	iter_counter = 0;
	error = 1.0;
	phase2(x, z, s); // runs Mehrotra's predictor-corrector algorithm on starting point (x,z,s)
	was_run = true;
}

int MVIntPointLP::smallest_ratio_index(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) const
{
	int index = -1;
	double val = -x(0)/dx(0);
	double tmp = 2.0;

	// find the index of smallest positive ratio -x(i)/dx(i) if all x(i) are always positive
	// in other words find the largest possible step alpha such that x + alpha * dx >= 0 for dx <= 0

	for (int i = 0; i < x.size(); i++)
	{
		tmp = -x(i) / dx(i);
		if (dx(i) < 0 && tmp <= val) {
			val = tmp;
			index = i;
		}
	}
	return index;
}

void MVIntPointLP::phase1(Eigen::VectorXd& x, Eigen::VectorXd& z, Eigen::VectorXd& s)
{
	Eigen::MatrixXd B = (A * A_t).inverse();
	Eigen::VectorXd x_hat = A_t * (B * b); // find closest to the origin point satisfying Ax = b
	z = B * (A * c); // minimize 0.5 * s^T * s over(s,z) subject to A^T * z + s = c
	Eigen::VectorXd s_hat = c - A_t * z;
	Eigen::VectorXd ones = Eigen::VectorXd::Ones(x.size());
	double delta_x = std::max(-3.0 / 2.0 * x_hat.minCoeff(), 0.0); // find smallest negative element in x and s and scale it by -3/2
	double delta_s = std::max(-3.0 / 2.0 * s_hat.minCoeff(), 0.0);
	x_hat += delta_x * ones; // make x_hat and s_hat strictly positive elementwise
	s_hat += delta_s * ones;
	double delta_x_hat = 0.5 * x_hat.dot(s_hat) / s_hat.sum(); // average size of components of x_hat weighted with weights from s_hat
	double delta_s_hat = 0.5 * x_hat.dot(s_hat) / x_hat.sum();
	x = x_hat + delta_x_hat * ones; // set x and s to their weighted averages + non-negative x_hat, s_hat
	s = s_hat + delta_s_hat * ones;
}

void MVIntPointLP::phase2(Eigen::VectorXd& x, Eigen::VectorXd& z, Eigen::VectorXd& s)
{
	int m = (int)A.rows();
	int n = (int)A.cols();

	Eigen::MatrixXd KKT_Matrix(m + 2 * n, m + 2 * n);
	Eigen::VectorXd residual(m + 2 * n);
	Eigen::VectorXd solution(m + 2 * n);
	Eigen::PartialPivLU<Eigen::MatrixXd> dec(m + 2 * n);

	/*
	KKT Matrix = [0, A^T, I,
				  A,  0,  0,
				  S,  0,  X]
				 where S = diag(s), X = diag(x)
	*/
	KKT_Matrix.setZero();
	KKT_Matrix.block(0, n, n, m) = A_t;
	KKT_Matrix.block(0, n + m, n, n) = Eigen::MatrixXd::Identity(n, n);
	KKT_Matrix.block(n, 0, m, n) = A;

	double mu = 1.0; // surrogate duality gap
	double mu_aff = 1.0; // surrogate duality gap if we were to take step in the affine scaling direction
	double sigma = 0.9; // measures how aggressive the steps should be (0 <= sigma < 1), sigma -> 0 solves KKT conditions while sigma -> 1 improves centrality

	double alpha_aff_primal = 1.0; // lengths of the primal and dual affine scaling step (predictor)
	double alpha_aff_dual = 1.0;
	double alpha_primal = 1.0; // lengths of the primal and dual step (corrector)
	double alpha_dual = 1.0;
	double eta = 0.95; // primal-dual step scaling coefficient (0.9 <= eta < 1 and eta_k -> 1 as k -> infinity)
	int index_primal = -1; // index of the element in dx_affine which has smallest positive ratio in -x(i)/dx(i)
	int index_dual = -1; // index of the element in ds_affine which has smallest positive ratio in -s(i)/ds(i)

	while (error >= tol && iter_counter < max_iter)
	{
		// initialize KKT_Matrix with new diagonal matrices X, S
		KKT_Matrix.block(n + m, 0, n, n) = Eigen::MatrixXd(s.asDiagonal());
		KKT_Matrix.block(n + m, n + m, n, n) = Eigen::MatrixXd(x.asDiagonal());

		// initialize residual (right hand side) of the linear system with new residuals
		residual.block(0, 0, n, 1) = -(A_t * z + s - c);
		residual.block(n, 0, m, 1) = -(A * x - b);
		residual.block(n + m, 0, n, 1) = -(x.array() * s.array());

		dec.compute(KKT_Matrix); // compute LU decomposition of KKT matrix
		solution = dec.solve(residual); // affine scaling step
		
		// find index i of the element for which -x(i)/dx(i) is smallest and positive
		// if all dx(i) are positive (and x(i) are positive by construction) then index is equal to -1
		index_primal = smallest_ratio_index(x, solution.block(0, 0, n, 1)); // dx
		index_dual = smallest_ratio_index(s, solution.block(n + m, 0, n, 1)); // ds

		// if index is -1 use affine step = 1.0, otherwise use affine step equal to negative ratio at the index 
		alpha_aff_primal = std::min(1.0, (index_primal == -1) ? 1.0 : -x(index_primal) / solution(index_primal));
		alpha_aff_dual = std::min(1.0, (index_dual == -1) ? 1.0 : -s(index_dual) / solution(index_dual + n + m));
		index_primal = -1;
		index_dual = -1;

		// surrogate duality gap
		mu = x.dot(s) / n;

		// surrogate duality gap if we were to take affine scaling step
		mu_aff = (x + alpha_aff_primal * solution.block(0, 0, n, 1)).dot(s + alpha_aff_dual * solution.block(n + m, 0, n, 1)) / n;
		sigma = std::pow(mu_aff / mu, 3.0);

		// compute new last n rows of the residual to solve KKT system with different right hand side for corrector direction
		residual.block(n + m, 0, n, 1) = -(x.array() * s.array() + solution.block(0, 0, n, 1).array() * solution.block(n + m, 0, n, 1).array() - sigma * mu);
		solution = dec.solve(residual);

		eta = 1.0 - 1.0 / (10.0 + pow(iter_counter, 2.0)); // eta_k = 1 - 1/(10+k^2)
		alpha_primal = std::min(1.0, eta * alpha_aff_primal); // compute primal-dual corrector steps
		alpha_dual = std::min(1.0, eta * alpha_aff_dual);

		x += alpha_primal * solution.block(0, 0, n, 1); // make a step in the corrected direction
		z += alpha_dual * solution.block(n, 0, m, 1);
		s += alpha_dual * solution.block(n + m, 0, n, 1);

		error = mu; // measure error by surrogate duality gap

		++iter_counter;
		if (std::abs(error) >= 1e+20) { // if the error gets larger than 10^20 consider it a diverging situation and stop
			isDivergent = true;
			break;
		}
	}
	result = x; // copy answer to the problem to result and z,s to vector of dual variables
	dual_variables = Eigen::VectorXd(dimension + A.rows());
	dual_variables.block(0, 0, m, 1) = z;
	dual_variables.block(m, 0, n, 1) = s;
}

void MVIntPointLP::setStart(const Eigen::VectorXd& x, const Eigen::VectorXd& z, const Eigen::VectorXd& s)
{
	if (x.size() == dimension && z.size() == A.rows() && s.size() == dimension && x.minCoeff() >= 0 && s.minCoeff() >= 0) {
		// if x,z,s are of correct dimension and x,s are elementwise non-negative - they're valid starting points
		starting_point = Eigen::VectorXd(2 * dimension + A.rows());
		hasStart = true;
		was_run = false;
		starting_point.block(0, 0, dimension, 1) = x;
		starting_point.block(dimension, 0, A.rows(), 1) = z;
		starting_point.block(dimension + A.rows(), 0, dimension, 1) = s;
	}
	else throw 1;
}

void MVIntPointLP::setParams(double tol_, size_t max_iter_) 
{
	tol = std::max(1e-15, tol_);
	max_iter = std::max(2, max_iter);
	iter_counter = 0;
	was_run = false;
}

Eigen::VectorXd& MVIntPointLP::getDual()
{
	return dual_variables; // returns vector (z,s) of dual variables
}

bool MVIntPointLP::isDiverging() const
{
	return isDivergent;
}
