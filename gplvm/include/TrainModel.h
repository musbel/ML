
#ifndef C_ML_INCLUDE_TRAINMODEL_H_
#define C_ML_INCLUDE_TRAINMODEL_H_

#include "Model.h"


namespace ML
{

class TrainGplvm : public TrainVisitor
{
public:
	TrainGplvm() : numIterations(0), stepSize(0.001) {}
	typedef std::shared_ptr<TrainGplvm> Ptr;

	virtual ~TrainGplvm() {}

	virtual double likelihood(const std::vector<double> &x, std::vector<double> &grad);
	virtual int train(LatentModel::Ptr model, const Matrix& Y, Matrix& X);

	double getStepSize() const { return stepSize; }

private:
	LatentModel::Ptr model;
	Matrix Y;
	Matrix X;
	Matrix featureWeights;

	int numIterations;
	double stepSize;
};


class TrainGplvmWithBc : public TrainVisitor
{
public:
	TrainGplvmWithBc(double gamma = 0.0) : gamma(gamma), numIterations(0)
	{}
	typedef std::shared_ptr<TrainGplvmWithBc> Ptr;

	virtual ~TrainGplvmWithBc() {}

	virtual double likelihood(const std::vector<double> &x, std::vector<double> &grad);
	virtual int train(LatentModel::Ptr model, const Matrix& Y, Matrix& X);

	double getGamma() const { return gamma; }

private:
	void computeKernelMatrix(
		Matrix &kernel,
		const Matrix& X,
		const Matrix &weights,
		double gamma = 0.001
		) const;

	void computeGradientWrtData(
		std::vector<Matrix> &d_matA,
		const Matrix& matX,
		const Matrix& matY,
		const Matrix &mat_dK,
		const Matrix& W,
		const double gamma = 0.001
		) const;

private:
	LatentModel::Ptr model;
	Matrix Y;
	Matrix X;
	Matrix featureWeights;

	int numIterations;
	double gamma;
	Matrix backConstraintParams;
};

}

#endif /* C_ML_INCLUDE_TRAINMODEL_H_ */
