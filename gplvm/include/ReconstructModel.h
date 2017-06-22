
#ifndef C_ML_INCLUDE_RECONSTRUCTMODEL_H_
#define C_ML_INCLUDE_RECONSTRUCTMODEL_H_

#include "Model.h"


namespace ML
{

class ReconstructGplvm : public ReconstructVisitor
{
public:
	ReconstructGplvm() : activeSetSize(0) {}
	typedef std::shared_ptr<ReconstructGplvm> Ptr;

	virtual ~ReconstructGplvm() {}

	ML_API virtual std::vector<double> reconstruct(
		std::shared_ptr<LatentModel> model,
		const std::vector<double>& values,
		const std::vector<Constraint::Ptr>& constraints
		);

	ML_API virtual double likelihood(const std::vector<double>& x, std::vector<double>& grad);

	ML_API virtual void print();

private:
	double computeLikelihood(
		const Matrix& invK,
		const Matrix& Y,
		const Matrix& kS,
		const Matrix& pose,
		const Matrix& lat,
		const Matrix& Weights
		);

	// TODO: Refactor derivative calculations, kernel, etc.
	void derivativeWrtY(
		Matrix& dpose,
		const Matrix& invK,
		const Matrix& Y,
		const Matrix& kx,
		const Matrix& pose,
		const Matrix& weights = Matrix::Zero(0, 0)
		);
	void derivativeWrtLat(
		Matrix& dlat,
		const Matrix& invK,
		const Matrix& Y,
		const Matrix& X,
		const Matrix& kx,
		const Matrix& pose,
		const Matrix& lat,
		const Matrix& weights = Matrix::Zero(0, 0)
		);

private:
	LatentModel::Ptr model;
	Matrix Y;
	Matrix X;
	Vector meanY;
	std::vector<Constraint::Ptr> constraints;

	std::vector<bool> XY_order; //false=latent parameter, true=pose feature
	int callIndex;
	int activeSetSize;
	double closestPoseErr;
	int closestInd;
	double lastclosestPoseLikelihood;
	double closestPoseLikelihood;
	std::vector<int> activeSet;

	Matrix invK;
	Matrix K;

	Matrix featureWeights;	// TODO: Doesn't seem to be used!
};

}

#endif /* C_ML_INCLUDE_RECONSTRUCTMODEL_H_ */
