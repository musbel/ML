#include "RbfKernel.h"

#define NOISE_OFFSET 4.0
#define NOISE_SCALE 0.01
#define NOISE_SCALESQ NOISE_SCALE*NOISE_SCALE


namespace ML
{

double getNoiseHyperPrior(const double val)
{
	double b = val - NOISE_OFFSET;
	return (0.5 * NOISE_SCALESQ * b * b);
}

double getNoiseHyperPriorDerivative(double val)
{
	return (NOISE_SCALESQ * val);
}

double RbfKernel::computeVariance(const Matrix& invK, const Matrix& kx) const
{
	double LL = (kx * invK * kx.transpose())(0, 0);
	double var = exp_par[0](0, 0) + 1.0 / exp_par[2](0, 0) - LL;
	return var;
}

void RbfKernel::computeKernelMatrix(Matrix& K, const Matrix& X) const
{
	// 2*sigma_f*[(x1-x2)*M*(x1-x2)']
	K = Matrix::Zero(X.rows(), X.rows());

	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = i; j < X.rows(); j++)
		{
			double kv = 0.0;
			Matrix x_ij = (X.row(i) - X.row(j));
			Matrix inside;
			if (exp_par[1].rows() == 1)
				inside = exp_par[1] * x_ij * x_ij.transpose();
			else
				inside = x_ij * exp_par[1] * x_ij.transpose();

			if (i == j)
				kv = exp_par[0](0, 0) * exp(-0.5 * inside(0, 0)) + 1.0 / std::pow(exp_par[2](0, 0), 1.0);
			else
				kv = exp_par[0](0, 0) * exp(-0.5 * inside(0, 0));

			K(i, j) = K(j, i) = kv;
		}
	}
}

void RbfKernel::computeGradientWrtKernelParams(const Matrix& X, std::vector<Matrix>& dt) const
{
	dt = std::vector<Matrix>(parameters.size(), Matrix::Zero(X.rows(), X.rows()));
	dt[1] = Matrix::Zero(X.rows() * parameters[1].rows(), X.rows() * parameters[1].rows());
	Matrix inside;

	for (int p = 0; p < 2; p++)
	{
		Matrix d_p1;
		for (int i = 0; i < X.rows(); i++)
		{
			for (int j = 0; j < X.rows(); j++)
			{
				Matrix x_ij = X.row(i) - X.row(j);
				if (parameter_scales.size() == 0)
				{
					if (exp_par[1].rows() == 1)

						inside = exp_par[1] * x_ij * x_ij.transpose();
					else
						inside = x_ij * exp_par[1] * x_ij.transpose();
				}
				else
				{
					if (exp_par[1].rows() == 1)
						inside = parameter_scales[1](0, 0) * exp_par[1] * x_ij * x_ij.transpose();
					else
						inside = x_ij * (parameter_scales[1](0, 0) * exp_par[1]) * x_ij.transpose();
				}

				if (p == 0 && parameter_scales.size() == 0)
				{
					dt[p](i, j) = exp_par[0](0, 0) * exp(-0.5 * inside(0, 0));
				}
				else if (p == 0 && parameter_scales.size() != 0)
				{
					dt[p](i, j) = parameter_scales[0](0, 0) * exp_par[0](0, 0) * exp(-0.5 * inside(0, 0));
				}
				else if (p == 1)
				{
					//if it is there is just one distance parameter in rbf function, computations changes
					if (exp_par[1].rows() == 1)
					{
						double outside_val = (exp_par[1] * x_ij * x_ij.transpose())(0, 0);
						if (parameter_scales.size() == 0)
							dt[p](i, j) = -0.5 * outside_val * exp(-0.5 * inside(0, 0));
						else
							dt[p](i, j) = -0.5 * parameter_scales[1](0, 0) * outside_val * exp(-0.5 * inside(0, 0));

					}
					else
					{
						for (int r = 0; r < exp_par[1].rows(); r++)
						{
							double outside_val = exp_par[1](r, r) * x_ij(0, r) * x_ij(0, r);
							if (parameter_scales.size() == 0)
								dt[p](r * X.rows() + i, r * X.rows() + j) = -0.5 * outside_val * exp(-0.5 * inside(0, 0));
							else
								dt[p](r * X.rows() + i, r * X.rows() + j) = -0.5 * parameter_scales[1](0, 0) * outside_val
										* exp(-0.5 * inside(0, 0));
						}
					}
				}
			}
		}
	}

	if (parameter_scales.size() == 0)
		dt[2].diagonal() = Vector::Constant(X.rows(), -1.0 / std::pow(exp_par[2](0, 0), 1.0)); //KERNEL_WEIGHT*Vector::Constant(X.rows(), KERNEL_WEIGHT);
	else
		dt[2].diagonal() = Vector::Constant(X.rows(), (-1.0 / std::pow(exp_par[2](0, 0), 1.0)) * parameter_scales[2](0, 0));
}

void RbfKernel::computeGradientWrtParams(const Matrix& X, const Matrix &dL_dK, std::vector<Matrix>& dt) const
{
	std::vector<Matrix> vKernel_dt;
	std::vector<Matrix> vinvKernel;
	std::vector<Matrix> vAlpha;
	std::vector<double> vTraces;
	std::vector<Matrix> all_params;

	computeGradientWrtKernelParams(X, vKernel_dt);

	all_params.insert(all_params.end(), exp_par.begin(), exp_par.end());
//	all_params.insert(all_params.end(), parameters.begin(), parameters.end());

	dt.clear();
	for (int i = 0; i < vKernel_dt.size(); i++)
		dt.push_back(Matrix::Zero(parameters[i].rows(), parameters[i].rows()));

	for (int i = 0; i < dt.size(); i++)
	{
		Matrix theta_i_dt = parameters[i];
		if (i == 2)
		{
			theta_i_dt(0, 0) = getNoiseHyperPriorDerivative(theta_i_dt(0, 0));
		}

		if (i != 1)
		{
			dt[i](0, 0) = (dL_dK.array() * vKernel_dt[i].array()).sum() + theta_i_dt(0, 0);
		}
		else
		{
			int nSamples = X.rows();
			int nLat = parameters[i].rows();
			if (parameters[i].rows() != 1)
			{
				for (int r = 0; r < nLat; r++)
				{
					Matrix scale_r = vKernel_dt[i].block(r * nSamples, r * nSamples, nSamples, nSamples);
					dt[i](r, r) = (dL_dK.array() * scale_r.array()).sum() + theta_i_dt(r, r);
				}
			}
			else
			{
				Matrix scale_r = vKernel_dt[i];
				dt[i](0, 0) = (dL_dK.array() * scale_r.array()).sum() + theta_i_dt(0, 0);
			}
		}
	}

	//BOOST_LOG_TRIVIAL(debug) << "dt_params= " << dt[0] << "\n" << dt[1] << "\n" << dt[2] << std::endl;
	//BOOST_LOG_TRIVIAL(debug) << "Finished the part_derivatives for RBF parameters " << std::endl;
}

void RbfKernel::updateGradientWrtParams(std::vector<double>& grad, const Matrix& X, const Matrix &dL_dK, double alpha) const
{
	std::vector<Matrix> param_dt;
	computeGradientWrtParams(X, dL_dK, param_dt);

	for (int i = 0; i < param_dt.size(); i++)
	{
		const Matrix params = param_dt[i];
		for (int r = 0; r < params.rows(); r++)
		{
			if (i != 1)
				grad[i] = params(0, 0) * alpha;
			else
				grad[i] = params(r, r) * alpha;
		}
	}
}

Matrix RbfKernel::computeGradientWrtRbfLatentVars(const Matrix &X, int ind1, int ind2) const
{
	// In the first row derivative wrt x_ind1, in the second row derivative wrt x_ind2
	Matrix dt = Matrix::Zero(X.cols(), 2);
	int par_convert = 1; //exp

	Matrix par_0, par_1;
	if (par_convert == 0)
	{
	}
	else if (par_convert == 1)
	{
		par_0 = exp_par[0];
		par_1 = exp_par[1];
	}
	else
	{
		par_0 = parameters[0];
		par_1 = parameters[1];
	}

	for (int i = 0; i < X.cols(); i++)
	{
		//thinking that the second parameter of the rbf can be a diagonal matrix
		Matrix x_ij = X.row(ind1) - X.row(ind2);
		Matrix inside;
		if (par_1.rows() == 1)
		{
			inside = par_1 * x_ij * x_ij.transpose();
			dt(i, 0) = (-1.0) * par_0(0, 0) * par_1(0, 0) * (X(ind1, i) - X(ind2, i)) * exp(-0.5 * inside(0, 0));
			dt(i, 1) = par_0(0, 0) * par_1(0, 0) * (X(ind1, i) - X(ind2, i)) * exp(-0.5 * inside(0, 0));
		}
		else
		{
			inside = x_ij * par_1 * x_ij.transpose();
			dt(i, 0) = (-1.0) * par_0(0, 0) * par_1(i, i) * (X(ind1, i) - X(ind2, i)) * exp(-0.5 * inside(0, 0));
			dt(i, 1) = par_0(0, 0) * par_1(i, i) * (X(ind1, i) - X(ind2, i)) * exp(-0.5 * inside(0, 0));
		}
	}

	return dt;
}

void RbfKernel::computeGradientWrtData(const Matrix& X, std::vector<Matrix> &part1, Matrix &part2) const
{
	part1 = std::vector<Matrix>(X.cols() * 2, Matrix::Zero(X.rows(), X.rows()));

	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = 0; j < X.rows(); j++)
		{
			Matrix kij_euclid = computeGradientWrtRbfLatentVars(X, i, j);
			for (int c = 0; c < X.cols(); c++)
			{
				part1[c](i, j) += kij_euclid(c, 0); // derivative wrt to x_i
				part1[X.cols() + c](i, j) += kij_euclid(c, 1); // derivative wrt to x_j
			}
		}
	}
}

void RbfKernel::computeGradientWrtData(
	std::vector<Matrix>& dt,
	const Matrix& X,
	const Matrix &dL_dK,
	int first_d,
	int last_d
	) const
{
	// Y=NxD, X=NxM, W=DxD, K=NxN  (M << D)

	std::vector<Matrix> part1; // for each dimension in X
	Matrix part2; // X.rows x  X.cols
	computeGradientWrtData(X, part1, part2);

	//find the change for x for each dimension
	dt = std::vector<Matrix>(1, Matrix::Zero(X.rows(), X.cols() - last_d - 1));
	Matrix mult_matrix = Matrix::Zero(X.rows(), X.rows());
	for (int i = 0; i < X.rows(); i++)
	{
		for (int c = 0; c < X.cols(); c++)
		{
			if (c >= first_d && c <= last_d) //back constraints latent vars
				continue;

			mult_matrix = Matrix::Zero(X.rows(), X.rows());
			mult_matrix.row(i) = part1[c].row(i);
			mult_matrix.col(i) = part1[X.cols() + c].col(i);
			mult_matrix(i, i) = part1[c](i, i) + part1[X.cols() + c](i, i);
			dt[0](i, c - last_d - 1) = (dL_dK.array() * mult_matrix.array()).sum() + X(i, c);
		}
	}
}

void RbfKernel::computeKxMatrix(
	Matrix &kx,
	std::vector<int> &activeSet,
	const Matrix &X,
	const Matrix &estimatedLatent
	) const
{
	kx = Matrix::Zero(1, activeSet.size());
	for (int i = 0; i < activeSet.size(); i++)
	{
		int ni = activeSet[i];
		Matrix x_ij;
		if (estimatedLatent.rows() != 1)
			x_ij = estimatedLatent.transpose() - X.row(ni);
		else
			x_ij = estimatedLatent - X.row(ni);

		Matrix inside;

		if (exp_par[1].rows() == 1)
		{
			inside = x_ij * x_ij.transpose();
			inside = exp_par[1] * inside;
		}
		else
		{
			inside = x_ij * exp_par[1] * x_ij.transpose();
		}

		kx(0, i) = exp_par[0](0, 0) * exp(-0.5 * inside(0, 0));
	}
}

void RbfKernel::initialiseParameters()
{
	// Define the 3 (currently) kernel parameters
	//  - 1st: 1x1 alpha
	//  - 2nd: nxn beta, n >= 1
	//  - 3rd: 1x1 gamma
	if (diagonalMatrixDims > 1)
	{
		parameters = std::vector<Matrix>(3, Matrix::Zero(1, 1));
		parameters[0] = Matrix::Zero(1, 1);
		parameters[1] = Matrix::Zero(diagonalMatrixDims, diagonalMatrixDims);
		parameters[2] = Matrix::Zero(1, 1);
	}
	else
	{
		parameters = std::vector<Matrix>(3, Matrix::Zero(1, 1));
	}

	parameters[2](0, 0) = 0.01;

	// A trick to avoid negative kernel values during optimisation. We calculate the
	// exponential of the 1st and 3rd value, and set the diagonal of the 2nd exp matrix
	// to 1 (and the rest to 0).
	std::vector<Matrix> exp_param;
	EigenHelper::exp<Matrix>(parameters, exp_param);
	if (diagonalMatrixDims > 1)
	{
		exp_param[1] = Matrix::Zero(diagonalMatrixDims, diagonalMatrixDims);
		exp_param[1].diagonal() = Vector::Ones(diagonalMatrixDims);
	}
	else
		exp_param[1](0, 0) = 1.0;

	exp_param[2] = parameters[2].array().exp();
	this->exp_par = exp_param;

	// Calculate the number of kernel parameters needed.
	// For the 2nd nxn matrix we're only interested in the number of diagonals
	// (it's a square matrix so the row number is the correct number).
	numParams = 0;
	for (auto p = parameters.begin(); p != parameters.end(); ++p)
		numParams += p->rows();
}

void RbfKernel::updateParameters(const std::vector<double> &x)
{
	// Populate the parameters with the new estimated data
	int index = 0;
	for (int i = 0; i < parameters.size(); ++i)
	{
		for (int j = 0; j < parameters[i].rows(); j++)
			parameters[i](j, j) = x[index++];
	}

	// Calculate the exp values and zero the off-diagonal values on
	// the 2nd nxn matrix
	EigenHelper::exp<Matrix>(parameters, exp_par);
	EigenHelper::setOffDiagonal(exp_par[1], 0.0);
}

void RbfKernel::setParameters(const std::vector<Matrix>& p)
{
	parameters = p;
	EigenHelper::exp<Matrix>(parameters, exp_par);
	EigenHelper::setOffDiagonal(exp_par[1], 0.0);
}

double RbfKernel::getPrior() const
{
	double parPrior = 0;
	for (int par_it = 0; par_it < parameters.size(); par_it++)
	{
		if (par_it < 2)
		{
			parPrior += 0.5 * (parameters[par_it] * parameters[par_it].transpose()).sum();
		}
		else
		{
			parPrior += getNoiseHyperPrior(parameters[par_it](0, 0));
		}
	}

	return parPrior;
}

void RbfKernel::derivativeKx(
	Matrix& dt,
	const Matrix& kx,
	const Matrix& x,
	const Matrix& X
	) const
{
	Vector kx_vec = Eigen::Map<const Vector>(kx.data(), kx.size());
	Vector x_vec = Eigen::Map<const Vector>(x.data(), x.size());
	dt = Matrix::Zero(X.rows(), X.cols());

	for (int r = 0; r < X.rows(); r++)
	{
		for (int c = 0; c < X.cols(); c++)
		{
			// -beta * (x - x') * k(x, x') [1]:(20)
			dt(r, c) = -exp_par[1](0, 0) * (x_vec[c] - X(r, c)) * kx_vec[r];
		}
	}
}

}
