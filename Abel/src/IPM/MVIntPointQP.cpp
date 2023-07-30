#include "pch.h"
#include "IPM/MVIntPointQP.h"

MVIntPointQP::MVIntPointQP(const Eigen::MatrixXd& G, const Eigen::VectorXd& c, const Eigen::MatrixXd& A, 
	const Eigen::VectorXd& b, const Eigen::MatrixXd& B, const Eigen::VectorXd& d, size_t dimension, double tol, int max_iter) 
	: G(G), B(B), B_t(B.transpose()), d(d), MVIntPointLP(c,A,b,dimension,tol,max_iter) {
	if (G.rows() != dimension || G.cols() != dimension || B.cols() != dimension || d.rows() != B.rows()) throw 1;
}

MVIntPointQP::MVIntPointQP(const MVIntPointQP& ip) : G(ip.G), B(ip.B), B_t(ip.B_t), d(ip.d), MVIntPointLP(ip) {}

MVIntPointQP& MVIntPointQP::operator=(const MVIntPointQP& ip)
{
	MVIntPointLP::operator=(ip);

	G = ip.G;
	B = ip.B;
	B_t = ip.B_t;
	d = ip.d;

	return *this;
}

// minimize 0.5 * x^T G x + c^T x subject to Ax=b, Bx<=d
// This transforms constraints into Ax=b, Bx+y=d, y >= 0
// used convention of variables is (x,l,z,y) where (x,y) are primal, (l,z) are dual variables with l >= 0
// Jorge Nocedal, Stephen Wright Numerical Optimization p. 484 -- Predictor-Corrector Algorithm for QP
void MVIntPointQP::solve() noexcept
{
	if (was_run) return;

	int m = (int)B.rows();
	int n = (int)B.cols();
	int k = (int)A.rows();
	int p = (int)A.cols();

	Eigen::VectorXd x(dimension);
	Eigen::VectorXd l(m);
	Eigen::VectorXd y(m);
	Eigen::VectorXd z(k);
	if (hasStart) { // use starting_point if given
		x = starting_point.block(0, 0, dimension, 1);
		l = starting_point.block(dimension, 0, m, 1);
		z = starting_point.block(dimension + m, 0, k, 1);
		y = starting_point.block(dimension + m + k, 0, m, 1);
	}
	else
	{
		x.setRandom();
		l.setConstant(1.0);
		z.setRandom();
		y.setConstant(1.0);
	}
	iter_counter = -1;
	error = 1.0;

	/*
	KKT_Matrix = [G, B^T, A^T, 0, (dx)
				  B,  0,   0,  I, (dl)
				  A,  0,   0,  0, (dz)
				  0,  Y,   0,  L] (dy)

	where Y = diag(y), L = diag(L), I = identity
	*/

	Eigen::MatrixXd KKT_Matrix(dimension + 2 * m + k, dimension + 2 * m + k);
	Eigen::VectorXd residual(dimension + 2 * m + k);
	Eigen::VectorXd solution(dimension + 2 * m + k);
	Eigen::PartialPivLU<Eigen::MatrixXd> dec(dimension + 2 * m + k);

	KKT_Matrix.setZero();
	KKT_Matrix.block(0, 0, dimension, dimension) = G;
	KKT_Matrix.block(0, dimension, n, m) = B_t;
	KKT_Matrix.block(0, dimension + m, p, k) = A_t;
	KKT_Matrix.block(dimension, 0, m, n) = B;
	KKT_Matrix.block(dimension + m, 0, k, p) = A;

	KKT_Matrix.block(dimension, dimension + m + k, m, m) = Eigen::MatrixXd::Identity(m, m);

	// complementarity measure mu, step alpha, step in the affine direction alpha_aff, 
	// primal and dual steps and coefficient sigma to measure how well should KKT system be approximated
	double mu, mu_aff, alpha, alpha_aff, alpha_primal, alpha_dual, sigma = 1.0;
	double tau = 0.9; // tau_k determines how much progress should be made after step

	int index_primal = -1;
	int index_dual = -1;

	while (error >= tol && iter_counter < max_iter)
	{
		KKT_Matrix.block(dimension + m + k, dimension, m, m) = y.asDiagonal();
		KKT_Matrix.block(dimension + m + k, dimension + m + k, m, m) = l.asDiagonal(); // construct KKT matrix and residual right hand side

		residual.block(0, 0, dimension, 1) = -(G * x + c + B_t * l + A_t * z);
		residual.block(dimension, 0, m, 1) = -(B * x - d + y);
		residual.block(dimension + m, 0, k, 1) = -(A * x - b);
		residual.block(dimension + m + k, 0, m, 1) = -(y.array() * l.array()); // compute affine scaling direction
		if (residual.norm() >= 1e+10) { isDivergent = true; return; } // detect divergence if residual gets too high

		dec.compute(KKT_Matrix);
		solution = dec.solve(residual); // compute affine scaling (predictor) step

		if (iter_counter == -1) { // phase 1 to make a heuristic guess for y and l, uses affine scaling direction
			for (size_t i = 0; i < m; i++)
			{
				y(i) = std::max(1.0, std::abs(y(i) + solution(dimension + m + k + i)));
				l(i) = std::max(1.0, std::abs(l(i) + solution(dimension + i)));
			}
			++iter_counter;
			continue;
		}
		else {
			// compute indices of elements with smallest allowed step length before violating nonegativity
			index_primal = smallest_ratio_index(y, solution.block(dimension + m + k, 0, m, 1));
			index_dual = smallest_ratio_index(l, solution.block(dimension, 0, m, 1));

			// if both allowed primal and dual steps are bounded by the ratio, use smallest
			// otherwise use the one that's bounded or use step = 1.0 if both are unbounded
			// affine step should be no larger than 1
			if (index_primal != -1 && index_dual != -1)
				alpha_aff = std::min(-y(index_primal) / solution(dimension + m + k + index_primal), -l(index_dual) / solution(dimension + index_dual));
			else if (index_primal == -1 && index_dual != -1)
				alpha_aff = -l(index_dual) / solution(dimension + index_dual);
			else if (index_primal != -1 && index_dual == -1)
				alpha_aff = -y(index_primal) / solution(dimension + m + k + index_primal);
			else
				alpha_aff = 1.0;
			alpha_aff = std::min(1.0, alpha_aff);

			mu = y.dot(l) / m; // compute current and prognosed complementary measure
			mu_aff = (y + alpha_aff * solution.block(dimension + m + k, 0, m, 1)).dot(l + alpha_aff * solution.block(dimension, 0, m, 1)) / m;
			sigma = pow(mu_aff / mu, 3.0);

			// change right hand side to solve for the corrector step
			residual.block(dimension + m + k, 0, m, 1) = -(y.array() * l.array() +
				solution.block(dimension + m + k, 0, m, 1).array() * solution.block(dimension, 0, m, 1).array() - sigma * mu);
			solution = dec.solve(residual);

			tau = 1.0 - 1.0 / (10.0 + pow(iter_counter, 2.0)); // tau_k = 1 - 1/(10 + k^2), tau_k -> 1 as k -> infinity

			// find steps alpha_primal and alpha_dual 
			// such that (y + alpha_primal * dy) >= (1-tau)*y and (l + alpha_dual * dl) >= (1-tau)*l
			index_primal = smallest_ratio_index(y, solution.block(dimension + m + k, 0, m, 1));
			index_dual = smallest_ratio_index(l, solution.block(dimension, 0, m, 1));
			alpha_primal = (index_primal == -1) ? 1.0 : -tau * y(index_primal) / solution(dimension + m + k + index_primal);
			alpha_dual = (index_dual == -1) ? 1.0 : -tau * l(index_dual) / solution(dimension + index_dual);

			alpha = std::min(alpha_primal, alpha_dual); // choose smallest allowed step and no larger than 1.0 as common step
			alpha = std::min(1.0, alpha);
			x += alpha * solution.block(0, 0, dimension, 1);
			l += alpha * solution.block(dimension, 0, m, 1);
			z += alpha * solution.block(dimension + m, 0, k, 1);
			y += alpha * solution.block(dimension + m + k, 0, m, 1);

			error = mu;
			++iter_counter;
		}
	}

	result = x;
	dual_variables = Eigen::VectorXd(2 * m + k); // (x, l, z, y)
	dual_variables.block(0, 0, m, 1) = l;
	dual_variables.block(m, 0, k, 1) = z;
	dual_variables.block(m + k, 0, m, 1) = y;
	was_run = true;
}

void MVIntPointQP::setStart(const Eigen::VectorXd& x, const Eigen::VectorXd& l, const Eigen::VectorXd& z, const Eigen::VectorXd& y)
{
	// (x,lambda,z,y)
	int A_rows = (int)A.rows();
	int B_rows = (int)B.rows();
	if (l.minCoeff() < 0 || y.minCoeff() < 0) throw 1;
	if (x.rows() != dimension 
		|| l.rows() != B_rows
		|| y.rows() != B_rows
		|| z.rows() != A_rows) throw 1;

	starting_point = Eigen::VectorXd(dimension + 2 * B_rows + A_rows);
	starting_point.block(0, 0, dimension, 1) = x;
	starting_point.block(dimension, 0, B_rows, 1) = l;
	starting_point.block(dimension + B_rows, 0, A_rows, 1) = z;
	starting_point.block(dimension + B_rows + A_rows, 0, B_rows, 1) = y;
	was_run = false;
	hasStart = true;
}