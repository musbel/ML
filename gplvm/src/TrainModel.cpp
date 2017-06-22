#include "TrainModel.h"

#include <iostream>
#include <cctype>
#include <algorithm>
#include <utility>
#include <math.h>
#include <float.h>
#include <random>
#include <exception>

#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>


namespace ML
{

double TrainGplvm::likelihood(const std::vector<double> &x, std::vector<double> &grad)
{
	kernel->updateParameters(x);
	int numLatDims = model->getNumOfLatentDims();

	double regulator = numIterations < 50 ? 0.001 : 1.0;
	grad = std::vector<double>(x.size(), 0.0);

	int index = kernel->getNumParams();	// The latent values start after the kernel parameters
	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = 0; j < X.cols(); j++)
			X(i, j) = x[index++];
	}

	// TODO: Question
	// Should the dL/dK be a part of the kernel or is it model specific?
	// If so, should all of the following be in the kernel (until likelihood calc.)
	// If not, then it's perhaps best to extract the following into a function
	// called e.g. kernelDerivativeWrtLikelihood.
	Matrix K_eucledian;
	kernel->computeKernelMatrix(K_eucledian, X);

	Matrix K_sum = K_eucledian;
	Matrix L = K_sum.llt().matrixL();

	Matrix LinvY = K_sum.llt().matrixL().solve(Y);
	Matrix KinvY = K_sum.llt().matrixL().transpose().solve(LinvY);
	Matrix K_sum_inv = K_sum.llt().solve(Matrix::Identity(Y.rows(), Y.rows()));
	Matrix dL_dK = -0.5 * (KinvY * KinvY.transpose() - Y.cols() * K_sum_inv);

	kernel->updateGradientWrtParams(grad, X, dL_dK, regulator);

	std::vector<Matrix> data_dt;
	kernel->computeGradientWrtData(data_dt, X, dL_dK, -1, -1);

	// TODO: Should this be in the kernel as well?
	for (int m = 0; m < data_dt.size(); m++)
	{
		for (int r = 0; r < data_dt[m].rows(); r++)
		{
			for (int c = 0; c < data_dt[m].cols(); c++)
				grad[index++] = data_dt[m](r, c);
		}
	}

	// Compute the likelihood
	double trace_Kinv_YYt = LinvY.squaredNorm();
	double logDet = 2.0 * (L.diagonal().array().log()).sum();

	double logLikelihood = (0.5 * Y.cols() * Y.rows()) * std::log(2.0 * M_PI);
	logLikelihood += Y.cols() * 0.5 * logDet;
	logLikelihood += 0.5 * trace_Kinv_YYt;
	logLikelihood += kernel->getPrior() + 0.5 * X.squaredNorm();

	std::cout << "> (GPLVM) Iteration " << numIterations << ": likelihood = " << logLikelihood << std::endl;
	numIterations++;
	return logLikelihood;
}

int TrainGplvm::train(LatentModel::Ptr model, const Matrix& Y, Matrix& X)
{
	int numLatDims = model->getNumOfLatentDims();
	kernel = model->getKernel();
	this->model = model;
	this->Y = Y;
	this->X = X;
	numIterations = 0;

	kernel->initialiseParameters();
	int numKernelParams = kernel->getNumParams();

	// The vecX is used by the optimisers
	int problemSize = numKernelParams + X.size();
	std::vector<double> vecX(problemSize);

	// Copy the kernel values to the optimiser input vector
	std::vector<Matrix> kernelParams = kernel->getParameters();
	int index = 0;
	for (int i = 0; i < kernelParams.size(); i++)
	{
		for (int j = 0; j < kernelParams[i].rows(); j++)
			vecX[index++] = kernelParams[i](j, j);
	}

	for (int i = 0; i < Y.rows(); i++)
	{
		for (int j = 0; j < numLatDims; j++)
			vecX[i * numLatDims + j + numKernelParams] = X(i, j);
	}

	if (optimiseStrategy->train(vecX, this))
	{
		// Copy estimated kernel parameters from optimiser vector
		for (int i = 0; i < kernelParams.size(); i++)
		{
			for (int j = 0; j < numLatDims; j++)
				kernelParams[i](j, j) = vecX[i * numLatDims + j + numKernelParams];
		}
		kernel->setParameters(kernelParams);

		// Copy the latent variables
		for (int i = 0; i < Y.rows(); i++)
		{
			for (int j = 0; j < numLatDims; j++)
			{
				X(i, j) = vecX[i * numLatDims + j + numKernelParams];
			}
		}
	}

	return 1;
}

double TrainGplvmWithBc::likelihood(const std::vector<double> &x, std::vector<double> &grad)
{
	kernel->updateParameters(x);
	int numLatDims = model->getNumOfLatentDims();

	double regulator = numIterations < 50 ? 0.001 : 1.0;
	grad = std::vector<double>(x.size(), 0.0);

	Matrix K;
	this->computeKernelMatrix(K, Y, featureWeights, gamma);

	Matrix matA = Matrix::Zero(Y.rows(), numLatDims);
	int index = kernel->getNumParams();	// The latent values start after the kernel parameters
	for (int i = 0; i < matA.rows(); i++)
	{
		for (int j = 0; j < matA.cols(); j++)
			matA(i, j) = x[index++];
	}

	X = K * matA;

	// TODO: Question
	// Should the dL/dK be a part of the kernel or is it model specific?
	// If so, should all of the following be in the kernel (until likelihood calc.)
	Matrix K_eucledian;
	kernel->computeKernelMatrix(K_eucledian, X);

	Matrix K_sum = K_eucledian;
	Matrix L = K_sum.llt().matrixL();

	Matrix LinvY = K_sum.llt().matrixL().solve(Y);
	Matrix KinvY = K_sum.llt().matrixL().transpose().solve(LinvY);
	Matrix K_sum_inv = K_sum.llt().solve(Matrix::Identity(Y.rows(), Y.rows()));
	Matrix dL_dK = -0.5 * (KinvY * KinvY.transpose() - Y.cols() * K_sum_inv);

	kernel->updateGradientWrtParams(grad, X, dL_dK, regulator);

	std::vector<Matrix> d_matA;
	this->computeGradientWrtData(d_matA, X, Y, dL_dK, featureWeights, gamma);

	int gradIdx = kernel->getNumParams();
	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = 0; j < X.cols(); j++)
			grad[gradIdx++] = d_matA[j].row(i).sum();
	}

	// Compute the likelihood
	double trace_Kinv_YYt = LinvY.squaredNorm();
	double logDet = 2.0 * (L.diagonal().array().log()).sum();

	double logLikelihood = (0.5 * Y.cols() * Y.rows()) * std::log(2.0 * M_PI);
	logLikelihood += Y.cols() * 0.5 * logDet;
	logLikelihood += 0.5 * trace_Kinv_YYt;
	logLikelihood += kernel->getPrior() + 0.5 * X.squaredNorm();

	std::cout << "> (GPLVM-BC) Iteration " << numIterations << ": likelihood = " << logLikelihood << std::endl;
	numIterations++;
	return logLikelihood;
}

int TrainGplvmWithBc::train(LatentModel::Ptr model, const Matrix& Y, Matrix& X)
{
	kernel = model->getKernel();
	int numLatDims = model->getNumOfLatentDims();
	this->model = model;
	this->Y = Y;
	this->X = X;
	numIterations = 0;

	kernel->initialiseParameters();
	int numKernelParams = kernel->getNumParams();

	// Compute the covariance matrix K and solve the back constraints
	// parameters from the latent variables
	Matrix K;
	this->computeKernelMatrix(K, Y, featureWeights, gamma);
	backConstraintParams = K.fullPivLu().solve(X);

	// The vecX is used by the optimisers
	int problemSize = numKernelParams + Y.rows() * numLatDims;
	std::vector<double> vecX(problemSize);

	// Copy the kernel values to the optimiser input vector
	std::vector<Matrix> kernelParams = kernel->getParameters();
	int index = 0;
	for (int i = 0; i < kernelParams.size(); i++)
	{
		for (int j = 0; j < kernelParams[i].rows(); j++)
			vecX[index++] = kernelParams[i](j, j);
	}

	// Copy the back constraint variables to the input vector
	for (int i = 0; i < Y.rows(); i++)
	{
		for (int j = 0; j < numLatDims; j++)
			vecX[i * numLatDims + j + numKernelParams] = backConstraintParams(i, j);
	}

	if (optimiseStrategy->train(vecX, this))
	{
		// Copying estimated back constraint parameters from optimiser vector
		for (int i = 0; i < Y.rows(); i++)
		{
			for (int j = 0; j < numLatDims; j++)
				backConstraintParams(i, j) = vecX[i * numLatDims + j + numKernelParams];
		}

		// Compute the latent variables
		X = K * backConstraintParams;
	}

	return 1;
}

void TrainGplvmWithBc::computeGradientWrtData(
	std::vector<Matrix> &d_matA,
	const Matrix& matX,
	const Matrix& matY,
	const Matrix &mat_dL_dK,
	const Matrix &W,
	const double gamma
	) const
{
	std::vector<Matrix> dt;
	d_matA = std::vector<Matrix>(matX.cols(), Matrix::Zero(matX.rows(), matX.rows()));
	kernel->computeGradientWrtData(dt, matX, mat_dL_dK, -1, -1);

	Matrix mat_dX = dt[0];
	for (int n = 0; n < matX.rows(); n++)
	{
		for (int m = 0; m < matX.rows(); m++)
		{
			for (int q = 0; q < matX.cols(); q++)
			{
				double kern_val = 0;
				Matrix diff = matY.row(n) - matY.row(m);
				if (W.cols() != 0 && W.rows() != 0)
				{
					Matrix rr = diff * W;
					kern_val = std::exp(-0.5 * gamma * rr.squaredNorm());
				}
				else
					kern_val = std::exp(-0.5 * gamma * (matY.row(n) - matY.row(m)).squaredNorm());

				d_matA[q](n, m) = mat_dX(m, q) * kern_val;
			}
		}
	}
}

void TrainGplvmWithBc::computeKernelMatrix(
	Matrix &kernel,
	const Matrix& Y,
	const Matrix &weight,
	double gamma
	) const
{
	kernel = Matrix::Zero(Y.rows(), Y.rows());
	for (int i = 0; i < Y.rows(); i++)
	{
		for (int j = i; j < Y.rows(); j++)
		{
			Matrix diff = (Y.row(i) - Y.row(j));
			if (weight.rows() != 0 && weight.cols() != 0)
			{
				Matrix t = diff * weight;
				Vector dist = Eigen::Map<Vector>(t.data(), t.size());
				kernel(i, j) = std::exp(-0.5 * gamma * dist.squaredNorm());
			}
			else
			{
				kernel(i, j) = std::exp(-0.5 * gamma * diff.squaredNorm());
			}
		}
	}

	kernel.triangularView<Eigen::Lower>() = kernel.transpose();
}

}
