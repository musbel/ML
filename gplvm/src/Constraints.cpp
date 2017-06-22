#include "Constraints.h"


namespace ML
{

double FeatureConstraint::synthesise(const std::vector<double>& x, std::vector<double>& grad)
{
	int idx = index + offset;
	grad = std::vector<double>(x.size(), 0.0);
	grad[idx] = std::pow(x[idx] - value, 1.0);
	double constraintValue = 0.5 * std::pow((x[idx] - value), 2.0);
	return constraintValue;
}

double BoneLengthConstraint::synthesise(const std::vector<double>& x, std::vector<double>& grad)
{
	// TODO: Make sure the latent dimensions have been set
	grad = std::vector<double>(x.size(), 0.0);
	double boneLength_sq = 0.0;

	for (int i = 0; i < set1.size(); i++)
	{
		int m = set1[i] + offset;
		int n = set2[i] + offset;
		boneLength_sq += std::pow((x[m] - x[n]), 2.0);
	}

	Matrix dt = Matrix::Zero(2, set1.size());
	for (int j = 0; j < set1.size(); j++)
	{
		int m = set1[j] + offset;
		int n = set2[j] + offset;
		dt(0, j) = 2.0 * (x[m] - x[n]);
		dt(1, j) = -2.0 * (x[m] - x[n]);
	}

	double lengthConstraint = boneLength_sq - value;

	for (int j = 0; j < set1.size(); j++)
	{

		int m = set1[j] + offset;
		int n = set2[j] + offset;

		grad[m] += dt(0, j);
		grad[n] += dt(1, j);
	}

	return lengthConstraint;
}

}
