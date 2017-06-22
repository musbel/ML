
#ifndef C_QUADRUPEDS_ML_INCLUDE_KERNEL_H_
#define C_QUADRUPEDS_ML_INCLUDE_KERNEL_H_

#include "EigenHelper.h"
#include "ExportHelper.h"

#include <iostream>
#include <memory>
#include <string>
#include <exception>
#include <vector>


namespace ML
{

class ML_API Kernel
{
protected:
	std::vector<Matrix> parameters;
	int numParams;

public:
	virtual ~Kernel() {}
	typedef std::shared_ptr<Kernel> Ptr;

	virtual void initialiseParameters() = 0;
	virtual void updateParameters(const std::vector<double>& x) = 0;
	virtual void setParameters(const std::vector<Matrix>& p) = 0;
	std::vector<Matrix> getParameters() const { return parameters; }

	virtual void computeKernelMatrix(Matrix& K, const Matrix& X) const = 0;

	virtual void computeGradientWrtParams(const Matrix& X, const Matrix& insideTrace, std::vector<Matrix>& dt) const = 0;
	virtual void updateGradientWrtParams(std::vector<double>& grad, const Matrix& X, const Matrix &dL_dK, double alpha) const = 0;

	virtual void computeGradientWrtData(
		std::vector<Matrix>& dt,
		const Matrix& X,
		const Matrix &insideTrace,
		int first_d = 0,
		int last_d = 1
		) const = 0;

	virtual double getPrior() const = 0;
	virtual double computeVariance(const Matrix& invK, const Matrix& kS) const = 0;

	int getNumParams() const { return numParams; }

	virtual void computeKxMatrix(
		Matrix& kx,
		std::vector<int>& activeSet,
		const Matrix& X,
		const Matrix& estimatedLatent
		) const = 0;

	virtual void derivativeKx(
		Matrix& dt,
		const Matrix& kx,
		const Matrix& x,
		const Matrix& X
		) const = 0;
};

} /* namespace ML */

#endif /* C_QUADRUPEDS_ML_INCLUDE_KERNEL_H_ */
