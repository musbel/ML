
#ifndef C_ML_INCLUDE_RBFKERNEL_H_
#define C_ML_INCLUDE_RBFKERNEL_H_

#include "Kernel.h"


namespace ML
{

class ML_API RbfKernel: public Kernel
{
public:
	RbfKernel(int diagonalMatrixDims = 1) : diagonalMatrixDims(diagonalMatrixDims)
	{
		numParams = 3;
		parameters = std::vector<Matrix>(numParams, Eigen::MatrixXd::Ones(1, 1));
		parameters[2](0, 0) = 0.1;

		if (diagonalMatrixDims != 1)
			parameters[1] = Eigen::MatrixXd::Zero(diagonalMatrixDims, diagonalMatrixDims);
	}
	virtual ~RbfKernel() {}
	typedef std::shared_ptr<RbfKernel> Ptr;

	virtual void initialiseParameters();
	virtual void updateParameters(const std::vector<double> &x);
	virtual void setParameters(const std::vector<Matrix>& p);

	virtual void computeKernelMatrix(Matrix& K, const Matrix& X) const;

//	virtual void computeGradientWrtLikelihood(const Matrix& X) const;

	virtual void computeGradientWrtParams(const Matrix& X, const Matrix &insideTrace, std::vector<Matrix>& dt) const;
	virtual void updateGradientWrtParams(std::vector<double>& grad, const Matrix& X, const Matrix &dL_dK, double alpha = 1.0) const;

	virtual double getPrior() const;
	virtual double computeVariance(const Matrix& invK, const Matrix& kS) const;

	virtual void computeGradientWrtData(
		std::vector<Matrix>& dt,
		const Matrix& X,
		const Matrix &insideTrace,
		int first_d = 0,
		int last_d = 1
		) const;

	virtual void computeKxMatrix(
		Matrix& kx,
		std::vector<int>& activeSet,
		const Matrix& X,
		const Matrix& estimatedLatent
		) const;

	virtual void derivativeKx(
		Matrix& dt,
		const Matrix& kx,
		const Matrix& x,
		const Matrix& X
		) const;

	void setDiagonalMatrixDims(int numDims);

private:
    Matrix computeGradientWrtRbfLatentVars(const Matrix &X, int ind1, int ind2) const;
	void computeGradientWrtKernelParams(const Matrix& X, std::vector<Matrix>& dt) const;
	void computeGradientWrtData(const Matrix& X, std::vector<Matrix> &part1, Matrix &part2) const;

private:
	int diagonalMatrixDims;
	std::vector<Matrix> exp_par;
	std::vector<Matrix> parameter_scales;
};

}

#endif /* C_ML_INCLUDE_RBFKERNEL_H_ */
