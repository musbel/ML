#include "Model.h"
#include "EigenHelper.h"


namespace ML
{

void CentreData::normalise(Matrix& Y)
{
	if (mean.rows() == 0)
	{
		EigenHelper::normaliseData<Matrix>(Y, Ynorm, mean, stdDev);
		Y = Ynorm;
	}
	else
	{
		// TODO: Can we do this more efficiently with Eigen?
		for (int i = 0; i < Y.rows(); ++i)
		{
			Y.row(i) = Y.row(i) - mean;
		}

		for (int i = 0; i < Y.cols(); ++i)
		{
			Y.col(i) = Y.col(i) / stdDev(i);
		}
	}
}

void CentreData::normalise(std::vector<double>& Y)
{
	for (int i = 0; i < Y.size(); ++i)
	{
		if (Y[i] == 0.0) continue;	// TODO: Implement
		Y[i] = Y[i] - mean(i);
	}

	for (int i = 0; i < Y.size(); ++i)
	{
		if (Y[i] == 0.0) continue;	// TODO: Implement
		Y[i] = Y[i] / stdDev(i);
	}
}

void CentreData::unnormalise(Matrix& Y)
{
	// TODO: Can we do this more efficiently with Eigen?
	//       E.g. Y = Y * stdDev + mean;
	for (int i = 0; i < Y.cols(); ++i)
	{
		Y.col(i) = Y.col(i) * stdDev(i);
	}

	for (int i = 0; i < Y.rows(); ++i)
	{
		Y.row(i) = Y.row(i) + mean;
	}
}

void CentreData::unnormalise(std::vector<double>& Y)
{
	for (int i = 0; i < Y.size(); ++i)
	{
		Y[i] = Y[i] * stdDev(i);
	}

	for (int i = 0; i < Y.size(); ++i)
	{
		Y[i] = Y[i] + mean(i);
	}
}

}
