#include "pch.h"
#include "IPM/MVIntPointSDP.h"

MVIntPointSDP::MVIntPointSDP(const std::vector<Eigen::MatrixXd*>& v, const Eigen::VectorXd& b, size_t dimension, double tol, size_t max_iter, bool hasLog) :
	v(v), b(&b), MVNumericalMethod(dimension, tol, max_iter, hasLog) {
	if (b.rows() > pow(dimension, 2) || b.rows() != v.size() - 1) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	if (hasLog) {
		lg = AbelLogger(11);
	}
}

MVIntPointSDP::MVIntPointSDP(const MVIntPointSDP& s) : v(s.v), X_start(s.X_start), primal_variable_X(s.primal_variable_X), dual_variable_S(s.dual_variable_S), 
dual_variable_y(s.dual_variable_y), b(s.b), isDivergent(s.isDivergent),
MVNumericalMethod(s.dimension, s.tol, s.max_iter, s.hasLog, s.iter_counter, s.error, Eigen::VectorXd(), Eigen::VectorXd(), s.hasStart, s.lg) {}

MVIntPointSDP& MVIntPointSDP::operator=(const MVIntPointSDP& s)
{
	v = s.v;
	X_start = s.X_start;
	primal_variable_X = s.primal_variable_X;
	dual_variable_S = s.dual_variable_S;
	dual_variable_y = s.dual_variable_y;
	b = s.b;
	isDivergent = s.isDivergent;
	MVNumericalMethod::operator=(s);

	return *this;
}

void MVIntPointSDP::solve() noexcept
{
	// On the Nesterov-Todd Direction in Semidefinite Programming -- Michael J. Todd, Kim-Chuan Toh, R. H. Tutuncu
	// DOI:10.1137/S105262349630060X
	// January 1997 SIAM Journal on Optimization 8(3)
	// https://www.researchgate.net/publication/2795874_On_the_Nesterov--Todd_Direction_in_Semidefinite_Programming
	// Mehrotra's Predictor-Corrector method with Nesterov-Todd direction usage for semidefinite programming problems in standard forms
	// minimize <C, X> subject to tr<A_i, X> = b_i, i=1,2,3,...,k and X >= 0 (positive semidefinite)

	int n = (int)((*v[0]).rows()); // dimension of the problem - working with n by n matrices
	int m = (*b).rows(); // number of equality constraints

	Eigen::MatrixXd X = (hasStart) ? X_start : Eigen::MatrixXd::Identity(n, n); // primal matrix
	Eigen::MatrixXd S = X; // matrix from the dual cone of PSD matrices
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
	Eigen::VectorXd y = Eigen::VectorXd::Ones(m); // vector of dual variables, the algorithm actually produces negative of true dual variables

	double mu = 1.0; // duality gap (X * S) / n;
	double mu_aff = 1.0; // predicted duality gap after affine scaling direction
	double sigma = 0; // centering parameter
	double alpha = 1.0; // step in the primal variables
	double beta = 1.0; // step in the dual variables
	double tau = 0.98; // defines how aggressive the steps are (0.99995 is value used in linear programming), closer to 1 -> more aggressive
	bool isFeasible = false;
	double primal_residual_norm = 0.0;
	double dual_residual_norm = 1.0;
	double primal_objective = 0;
	double dual_objective = 0;
	iter_counter = 0;

	Eigen::LLT<Eigen::MatrixXd> cholX(n); // cholesky of X = LL^T
	Eigen::LLT<Eigen::MatrixXd> cholS(n); // cholesky of S = RR^T = U^T * U, RR^T is actually used in Todd's paper
	Eigen::BDCSVD<Eigen::MatrixXd> svd(n, n, Eigen::DecompositionOptions::ComputeThinV); // svd of R^T * L

	Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n); // cholesky factor of X
	Eigen::MatrixXd L_inv = Eigen::MatrixXd::Zero(n, n); // inverse cholesky factor of X
	Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n); // cholesky factor of S
	Eigen::MatrixXd R_inv = Eigen::MatrixXd::Zero(n, n); // inverse cholesky factor of S

	Eigen::DiagonalMatrix<double, -1, -1> D; // matrix of singular values of R^T * L
	Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n); // orthogonal matrix V in SVD of R^T * L = UDV^T

	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n); // G = LVD^{-1/2} with property GG^T = W^{-1} where W is Nesterov-Todd scaling matrix, paper also used P = G^{-1}
	Eigen::MatrixXd G_inv = Eigen::MatrixXd::Zero(n, n); // inverse of G

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n * (n + 1) / 2); // matrix with rows svec(A_i) where A_i is matrix from constraint tr<A_i, X> = b_i
	Eigen::VectorXd X_r = Eigen::VectorXd::Zero(n * (n + 1) / 2); // vector used for adjustments to improve feasbility
	Eigen::MatrixXd smatX_r = Eigen::MatrixXd::Identity(n, n); // symmetric matrixization of X_r (inverse of symmetric vectorization)

	Eigen::MatrixXd R_d = Eigen::MatrixXd::Zero(n, n); // dual residual C - S - sum_{i=1}^m y_i A_i
	Eigen::VectorXd R_p = Eigen::VectorXd::Zero(m); // primal residual b - A * svec(X)

	Eigen::VectorXd h = Eigen::VectorXd::Zero(n * (n + 1) / 2); // right hand side in the least squares problem B_t * dy = h
	Eigen::MatrixXd B_t = Eigen::MatrixXd::Zero(n * (n + 1) / 2, m); // B_t (transposed) is a matrix with columns svec(G^T * A_i * G) where A_i are constraint matrices
	Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(n, n); // temporary storage matrix, used for computing right hand side h
	Eigen::HouseholderQR<Eigen::MatrixXd> qr(n * (n + 1) / 2, m); // qr factorization to solve least squares problem B_t * dy = h

	Eigen::VectorXd dy = Eigen::VectorXd::Zero(n); // changes in y
	Eigen::MatrixXd ds = Eigen::MatrixXd::Zero(n, n); // changes in S
	Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(n, n); // changes in X

	Eigen::MatrixXd R_nt = Eigen::MatrixXd::Zero(n, n); // Nesterov-Todd residual used as a correction term in second stage (corrector) of the algorithm
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(n); // eigenvalue decomposition to find smallest eigenvalues to choose steplengths
	Eigen::VectorXd svecX = Eigen::VectorXd::Zero(n * (n + 1) / 2); // symmetric vectorization of X

	// y vector being produced is negative of the true y-vector

	for (size_t i = 1; i < m + 1; i++)
	{
		svec(*v[i], A.block(i - 1, 0, 1, n * (n + 1) / 2)); // initialize matrix A
	}

	while (true) // main loop
	{
		mu = X.reshaped().dot(S.reshaped()) / n; // compute duality gap
		error = mu;

		cholX.compute(X);
		cholS.compute(S);

		L = cholX.matrixL();
		R = cholS.matrixU(); // paper uses R^T, here we're using decomposition S = U^T * U = RR^T and choose our R to be U to avoid transposes

		svd.compute(R * L.triangularView<Eigen::Lower>()); // compute svd of R^T * L

		D = svd.singularValues().asDiagonal();
		V = svd.matrixV();
		G = L.triangularView<Eigen::Lower>() * V * svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal(); // G = L * V * D^{-1/2}
		G_inv = svd.singularValues().cwiseSqrt().asDiagonal() * V.transpose() * (L.triangularView<Eigen::Lower>().solve(I)).triangularView<Eigen::Lower>(); // P

		svec(X, svecX);
		R_p = *b - A * svecX;
		R_d = *v[0] - S;
		for (size_t i = 1; i < m + 1; i++)
		{
			R_d -= y(i - 1) * *v[i];
		}
		primal_residual_norm = R_p.norm();
		dual_residual_norm = R_d.norm();

		isFeasible = (primal_residual_norm < tol) ? true : false; // if infeasibility is small enough, then the iterate is considered primal feasible

		if (!isFeasible) {
			AffineProjection(A, R_p, X_r); // find X_r such that AX_r = R_p by using least-norm solution
			smatX_r = smat(X_r, n);
			tmp = G.transpose() * R_d * G + G_inv * smatX_r * G_inv.transpose() + D.toDenseMatrix(); // formula (43) in Todd's paper with sigma = 0
		}
		else {
			tmp = G.transpose() * R_d * G + D.toDenseMatrix(); // if X is feasible, X_r = 0
		}

		svec(tmp, h);
		for (size_t i = 1; i < m + 1; i++)
		{
			svec(G.transpose() * *v[i] * G, B_t.col(i - 1)); // fill up matrix B^T
		}

		qr.compute(B_t);

		dy = qr.solve(h); // solve least squares problem B^T * dy = h
		ds = R_d - smat(A.transpose() * dy, n);
		if (isFeasible) {
			dx = -G * smat(h - B_t * dy, n) * G.transpose();
		}
		else {
			dx = smatX_r - G * smat(h - B_t * dy, n) * G.transpose();
		}

		L_inv = L.triangularView<Eigen::Lower>().solve(I); // compute inverses of triangular cholesky factors
		R_inv = R.triangularView<Eigen::Upper>().solve(I); // R was chosen to be U in U^T * U cholesky of S so it is upper triangular

		eig.compute(L_inv.triangularView<Eigen::Lower>() * dx * L_inv.transpose());
		alpha = std::min(1.0, -tau / eig.eigenvalues().minCoeff()); // choose alpha as min(1, -tau/lambda_min)

		eig.compute(R_inv.transpose() * ds * R_inv.triangularView<Eigen::Upper>()); // since R was chosen upper triangular the transposes are flipped from what paper describes
		beta = std::min(1.0, -tau / eig.eigenvalues().minCoeff()); // choose beta as min(1, -tau/lambda_min)

		// if the steps turn out to be negative, that means all eigenvalues are positive along computed direction, choose 1 in this case
		alpha = (alpha < 0) ? 1.0 : alpha;
		beta = (beta < 0) ? 1.0 : beta;

		// compute sigma as ratio of predicted complementarity <X+alpha*DX,S+beta*DS> and current complementarity <X,S>, all squared
		// use previously computed mu since mu = <X,S> / n;
		mu_aff = (X + alpha * dx).reshaped().dot((S + beta * ds).reshaped()) / n;
		sigma = pow(mu_aff * n, 2) / pow(mu * n, 2);
		sigma = (alpha + beta >= 1.8) ? std::max(0.05, sigma) : (alpha + beta >= 1.4) ? std::max(0.1, sigma) : std::max(0.2, sigma);
		// choose sigma depending on how fast the barrier parameter decreases

		R_nt = -(G_inv * dx * ds * G + G.transpose() * ds * dx * G_inv.transpose()) * D.inverse() / 2.0; // Nesterov-Todd second order correction in corrector step
		svec(tmp - sigma * mu * D.toDenseMatrix().inverse() - R_nt, h); // vectorize new right hand with Nesterov-Todd correction and non-zero value of sigma

		dy = qr.solve(h);
		ds = R_d - smat(A.transpose() * dy, n);
		if (isFeasible) {
			dx = -G * smat(h - B_t * dy, n) * G.transpose();
		}
		else {
			dx = smatX_r - G * smat(h - B_t * dy, n) * G.transpose();
		}

		eig.compute(L_inv.triangularView<Eigen::Lower>() * dx * L_inv.transpose());
		alpha = std::min(1.0, -tau / eig.eigenvalues().minCoeff());

		eig.compute(R_inv.transpose() * ds * R_inv.triangularView<Eigen::Upper>());
		beta = std::min(1.0, -tau / eig.eigenvalues().minCoeff());

		alpha = (alpha < 0) ? 1.0 : alpha;
		beta = (beta < 0) ? 1.0 : beta;

		if (hasLog) {
			primal_objective = v[0]->reshaped().dot(X.reshaped());
			dual_objective = y.dot(*b);
			lg.record(mu, 0);
			lg.record(primal_objective, 1);
			lg.record(dual_objective, 2);
			lg.record(alpha, 3);
			lg.record(beta, 4);
			lg.record(primal_residual_norm, 5);
			lg.record(dual_residual_norm, 6);
			lg.record(mu_aff, 7);
			lg.record(sigma, 8);
		}

		X += alpha * dx;
		S += beta * ds;
		y += beta * dy;

		++iter_counter;
		if (error > 1e+10 || std::max(primal_residual_norm,dual_residual_norm) > 1e+10) { isDivergent = true; break; }
		if (iter_counter > max_iter || (error < tol && std::max(primal_residual_norm, dual_residual_norm) < tol) || (sigma > 1) || (alpha < 1e-8 && beta < 1e-8)) { break; }

	}
	hasStart = false;
	if (isDivergent) { return; }
	primal_variable_X = X;
	dual_variable_S = S;
	dual_variable_y = -y; // y vector being produced by algorithm is negative of the true y-vector, so this stores actual dual variable
}

void MVIntPointSDP::setParams(double tol_, size_t max_iter_) noexcept
{
	tol = std::max(1e-15, tol_);
	max_iter = std::max((size_t)2, max_iter);
	iter_counter = 0;
}

void MVIntPointSDP::setStart(const Eigen::MatrixXd& X_)
{
	if (X_.rows() != dimension * dimension || X_.cols() != dimension * dimension) throw AbelException(ABEL_EX_MSG_INVALID_DIM, ABEL_EX_CODE_INVALID_DIM);
	X_start = X_;
	hasStart = true;
}

bool MVIntPointSDP::isDiverging() const noexcept
{
	return isDivergent;
}

void MVIntPointSDP::printLogs() const
{
	lg.print("MVIntPointSDP", { "Complementarity", "PrimalObjective", "DualObjective", "PrimalStep", "DualStep", "PrimalResidual", "DualResidual", "AffineComplementarity", "Sigma" });
}
