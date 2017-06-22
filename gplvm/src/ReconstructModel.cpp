#include "ReconstructModel.h"

#include <iostream>
#include <cctype>
#include <algorithm>
#include <numeric>
#include <utility>
#include <float.h>

#include <nlopt.hpp>
#include <boost/algorithm/string.hpp>


namespace ML
{

double ReconstructGplvm::computeLikelihood(
	const Matrix &invK,
	const Matrix &Y,
	const Matrix &kx,
	const Matrix &pose,
	const Matrix &lat,
	const Matrix &weights
	)
{
	Matrix fx = meanY.transpose() + (Y.transpose() * invK * kx.transpose()).transpose();

	double var = kernel->computeVariance(invK, kx);
	double first_part = (pose - fx).squaredNorm();
	return 0.5 * (first_part * (1.0 / var) + Y.cols() * std::log(var) + lat.squaredNorm());
}

double ReconstructGplvm::likelihood(const std::vector<double>& x, std::vector<double>& grad)
{
	double likelihood = 0.0;
	Vector pose;
	Matrix kx;
	std::vector<int> activeSet;
	activeSetSize = Y.rows();

	activeSet = std::vector<int>(activeSetSize, 0);
	for (int i = 0; i < activeSetSize; i++)
		activeSet[i] = i;

	Matrix estimatedLatent = Matrix::Zero(1, model->getNumOfLatentDims());
	Matrix estimatedFeatures = Matrix::Zero(1, Y.cols());

	int firstFalse = std::find(XY_order.begin(), XY_order.end(), false) - XY_order.begin();
	int firstTrue = std::find(XY_order.begin(), XY_order.end(), true) - XY_order.begin();

	for (int i = 0; i < XY_order.size(); i++)
	{
		if (XY_order[i] == false)
			estimatedLatent(0, i - firstFalse) = x[i - firstFalse];
		else
			estimatedFeatures(0, i - firstTrue) = x[i];
	}

	Vector dtPose;
	Matrix activeSetX = Matrix::Zero(activeSetSize, X.cols());
	Matrix activeSetY = Matrix::Zero(activeSetSize, Y.cols());

	for (int i = 0; i < activeSetX.rows(); i++)
	{
		activeSetX.row(i) = X.row(activeSet[i]);
		activeSetY.row(i) = Y.row(activeSet[i]);
	}

	if (invK.rows() != activeSetSize)
	{
		kernel->computeKernelMatrix(K, X);
		kernel->computeKxMatrix(kx, activeSet, X, estimatedLatent);
		invK = K.fullPivLu().solve(Matrix::Identity(activeSetSize, activeSetSize));
	}
	else
		kernel->computeKxMatrix(kx, activeSet, X, estimatedLatent);

	double firstLikelihood = likelihood = computeLikelihood(invK, activeSetY, kx, estimatedFeatures, estimatedLatent, featureWeights);

	// Get derivative wrt to Y
	Matrix dt_pose;
	Matrix dt_lat;

	derivativeWrtY(dt_pose, invK, activeSetY, kx, estimatedFeatures);
	derivativeWrtLat(dt_lat, invK, activeSetY, activeSetX, kx, estimatedFeatures, estimatedLatent);

	grad = std::vector<double>(XY_order.size(), 0.0);
	for (int i = 0; i < XY_order.size(); i++)
	{
		if (XY_order[i] == false)
			grad[i] = dt_lat(i - firstFalse);
		else
			grad[i] = dt_pose(i - firstTrue);
	}

	lastclosestPoseLikelihood = likelihood;
	callIndex++;

	return likelihood;
}

std::vector<double> toVec(Matrix& mat)
{
	std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
	return vec;
}

std::vector<double> ReconstructGplvm::reconstruct(
	std::shared_ptr<LatentModel> model,
	const std::vector<double>& values,
	const std::vector<Constraint::Ptr>& constraints
	)
{
	kernel = model->getKernel();
	this->model = model;
	Y = model->getY();
	std::vector<double> d_result;
	int numDims = Y.cols();
	int numTestSamples = 1; // TODO: int(d_values.size()) / int(b_values.size());

	int numLatentDims = model->getNumOfLatentDims();
	Matrix estLatent = Matrix::Zero(numTestSamples, numLatentDims);
	Matrix givenData = Matrix::Zero(numTestSamples, numDims);
	Matrix estData = Matrix::Zero(numTestSamples, numDims);
	Matrix matErr = Matrix::Zero(1, numTestSamples);
	Matrix matLikelihood = Matrix::Zero(1, numTestSamples);

//	featureWeights = Matrix::Identity(Y.cols(), Y.cols()); // ?

	X = model->getX();
	meanY = model->getMeanY();

	XY_order = std::vector<bool>(Y.cols() + X.cols(), false);
	for (int i = numLatentDims; i < XY_order.size(); i++)
	{
		XY_order[i] = true;
	}

	activeSetSize = Y.rows();

	// TODO: Verify that we have a kernel

	Matrix K;
	kernel->computeKernelMatrix(K, X);
	if (invK.rows() != activeSetSize)
	{
		invK = K.fullPivLu().solve(Matrix::Identity(activeSetSize, activeSetSize));
	}

	for (int i = 0; i < constraints.size(); ++i)
	{
		constraints[i]->setOffset(X.cols());
	}

	for (int sampleID = 0; sampleID < numTestSamples; ++sampleID)
	{
		callIndex = 0;
		closestPoseErr = DBL_MAX;
		closestInd = -1;

		int problemSize = X.cols() + numDims;
		std::vector<double> estimatedPose = std::vector<double>(problemSize, 0.0);

		Matrix X_mean = X.colwise().mean();

		for (int i = 0; i < X.cols(); i++)
			estimatedPose[i] = X_mean(0, i);

		for (int i = X.cols(); i < problemSize; i++)
			estimatedPose[i] = meanY[i - X.cols()];

		if (optimiseStrategy->reconstruct(estimatedPose, this, constraints))
		{
			for (int i = 0; i < problemSize; i++)
			{
				if (i < X.cols())
					estLatent(sampleID, i) = estimatedPose[i];
				else
				{
					givenData(sampleID, i - X.cols()) = values[sampleID * numDims + i - X.cols()];
					estData(sampleID, i - X.cols()) = estimatedPose[i];
				}
			}

			d_result.insert(d_result.end(), estimatedPose.begin() + numLatentDims, estimatedPose.end());
			matErr(0, sampleID) = closestPoseErr;
			matLikelihood(0, sampleID) = lastclosestPoseLikelihood;
		}
		else
		{
			std::cout << "Could not finish the optimisation for the missing features" << std::endl;

			for (int i = 0; i < problemSize; i++)
			{
				if (i < numLatentDims)
				{
					estLatent(sampleID, i) = estimatedPose[i];
				}
				else
				{
					givenData(sampleID, i - X.cols()) = values[sampleID * numDims + i - X.cols()];
					estData(sampleID, i - X.cols()) = estimatedPose[i];
				}
			}

			d_result.insert(d_result.end(), estimatedPose.begin() + numLatentDims, estimatedPose.end());
			matErr(0, sampleID) = closestPoseErr;
			matLikelihood(0, sampleID) = lastclosestPoseLikelihood;
		}
	}

	return d_result;
}

void ReconstructGplvm::print()
{
	std::cout << "### ReconstructGplvm ###" << std::endl;
	std::cout << " - Likelihood = " << closestPoseLikelihood << std::endl;
	std::cout << " - Likelihood (last) = " << lastclosestPoseLikelihood << std::endl;
	std::cout << " - Pose error = " << closestPoseErr << std::endl;
	std::cout << " - Call index = " << callIndex << std::endl;
	std::cout << " - Closest index = " << closestInd << std::endl;
}

// TODO: Refactor the derivative calculations, relationship with kernel, etc.
void ReconstructGplvm::derivativeWrtY(
	Matrix& dpose,
	const Matrix& invK,
	const Matrix& Y,
	const Matrix& kx,
	const Matrix& pose,
	const Matrix& weights
	)
{
	// f(x): The pose that the model would predict for a given x; this is equivalent to the
	//       RBF interpolation of the training poses
	// K:    The kernel matrix for the active set
	// Y:    The mean adjusted matrix of active set points
	// k(x): A vector in which the i-th entry contains k(x, x_i), i.e. the similarity
	//       between x and the i-th point in the active set

	Matrix fx = meanY.transpose() + (Y.transpose() * invK * kx.transpose()).transpose(); // [1]:(4)
	double var = kernel->computeVariance(invK, kx); // [1]:(5)

	Matrix first_part = Matrix::Zero(1, Y.cols());
	if (weights.rows() != Y.cols() && weights.cols() != Y.cols())
	{
		first_part = pose - fx;
	}
	else
	{
		first_part = (weights * weights.transpose() * (pose - fx));
	}

	dpose = first_part.array() * (1.0 / var);
}

// TODO:
// Question 1 - Isn't this dL_IK/dx?
// Question 2 - What's the difference between X and lat?
void ReconstructGplvm::derivativeWrtLat(
	Matrix& dlat,
	const Matrix& invK,
	const Matrix& Y,
	const Matrix& X,
	const Matrix& kx,
	const Matrix& pose,
	const Matrix& lat,
	const Matrix& weights
	)
{
	// dL_IK/dx - [1]:(17)
	Matrix fx = (meanY.transpose() + (Y.transpose() * invK * kx.transpose()).transpose());
	double var = kernel->computeVariance(invK, kx);

	double invVar = 1.0 / var;
	Matrix first_part = Matrix::Zero(1, Y.cols());
	double kk = 0.0;

	if (weights.rows() != Y.cols() && weights.cols() != Y.cols())
	{
		first_part.row(0) = (pose - fx).array() * invVar;
		kk = (Y.cols() - (pose - fx).squaredNorm() * invVar) * invVar * 0.5;
	}
	else
	{
		first_part.row(0) = (weights * weights.transpose() * (pose - fx));
		kk = (Y.cols() - ((weights * weights.transpose()) * (pose - fx)).squaredNorm() * invVar) * invVar * 0.5;
	}

	Matrix dt_kx;
	kernel->derivativeKx(dt_kx, kx, lat, X);

	Matrix dt_fx = Y.transpose() * invK * dt_kx; // [1]:(18)
	Matrix dt_var = -2.0 * kx * invK * dt_kx;	// [1]:(19)

	Matrix left = -1.0 * dt_fx.transpose() * first_part.transpose();
	Matrix right = dt_var * kk + lat;
	dlat = left.transpose() + right;
}

}

// References:
// [1] Style-Based Inverse Kinematics, Popovic, 2004
