
#ifndef C_QUADRUPEDS_ML_INCLUDE_MODEL_H_
#define C_QUADRUPEDS_ML_INCLUDE_MODEL_H_

#include <memory>

#include "RbfKernel.h"
#include "EigenHelper.h"
#include "ExportHelper.h"
#include "Constraints.h"

#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>

typedef std::shared_ptr<Matrix> MatrixPtr;
typedef std::shared_ptr<Vector> VectorPtr;
typedef std::vector<Matrix> Matrices;


namespace ML
{

class InitialiseStrategy
{
public:
	InitialiseStrategy() {}
	virtual ~InitialiseStrategy() {}
	typedef std::shared_ptr<InitialiseStrategy> Ptr;

	virtual void initialise(int numLatDims, Matrix& Y, Matrix& X, Vector& meanY, double& maxY) = 0;
};


class PcaInitialiser : public InitialiseStrategy
{
public:
	PcaInitialiser(bool normalise = false) : normalise(normalise) {}
	virtual ~PcaInitialiser() {}
	typedef std::shared_ptr<PcaInitialiser> Ptr;

	virtual void initialise(int numLatDims, Matrix& Y, Matrix& X, Vector& meanY, double& maxY);

	int getNumSamples() const { return numSamples; }

private:
	bool normalise;
	int numSamples;
};


class LatentModel;
class OptimiseStrategy;

class TrainVisitor
{
public:
	TrainVisitor() {}
	virtual ~TrainVisitor() {}
	typedef std::shared_ptr<TrainVisitor> Ptr;

	virtual int train(std::shared_ptr<LatentModel> model, const Matrix& Y, Matrix& X) = 0;
	virtual double likelihood(const std::vector<double> &x, std::vector<double> &grad) = 0;

	virtual void print() {}

	void setKernel(Kernel::Ptr kernel)
	{
		this->kernel = kernel;
	}

	void setOptimiseStrategy(std::shared_ptr<OptimiseStrategy> optimiseStrategy)
	{
		this->optimiseStrategy = optimiseStrategy;
	}

protected:
	Kernel::Ptr kernel;
	std::shared_ptr<OptimiseStrategy> optimiseStrategy;
};


class ReconstructVisitor
{
public:
	ReconstructVisitor() {}
	virtual ~ReconstructVisitor() {}
	typedef std::shared_ptr<ReconstructVisitor> Ptr;

	virtual std::vector<double> reconstruct(
			std::shared_ptr<LatentModel> model,
			const std::vector<double>& values,
			const std::vector<Constraint::Ptr>& constraints
			) = 0;

	virtual double likelihood(const std::vector<double>& x, std::vector<double>& grad) = 0;

	virtual void print() {}

	void setKernel(Kernel::Ptr kernel)
	{
		this->kernel = kernel;
	}

	void setOptimiseStrategy(std::shared_ptr<OptimiseStrategy> optimiseStrategy)
	{
		this->optimiseStrategy = optimiseStrategy;
	}

protected:
	Kernel::Ptr kernel;
	std::shared_ptr<OptimiseStrategy> optimiseStrategy;
};


class OptimiseStrategy
{
public:
	OptimiseStrategy() {}
	virtual ~OptimiseStrategy() {}
	typedef std::shared_ptr<OptimiseStrategy> Ptr;

	virtual int train(std::vector<double>& data, TrainVisitor* train) = 0;
	virtual int reconstruct(
		std::vector<double>& data,
		ReconstructVisitor* reconstruct,
		const std::vector<Constraint::Ptr>& constraints
		) = 0;
};


class NormaliseStrategy
{
public:
	NormaliseStrategy() {}
	virtual ~NormaliseStrategy() {}
	typedef std::shared_ptr<NormaliseStrategy> Ptr;

	virtual void normalise(Matrix& Y) = 0;
	virtual void normalise(std::vector<double>& Y) = 0;
	virtual void unnormalise(Matrix& Y) = 0;
	virtual void unnormalise(std::vector<double>& Y) = 0;

// TODO: Warning: This shouldn't be here! Late night fix.
public: // TODO: Temp until strategies learn how to load/save themselves
	Matrix mean;
	Matrix stdDev;
	Matrix Ynorm;
};


class CentreData : public NormaliseStrategy
{
public:
	CentreData() {}
	virtual ~CentreData() {}
	typedef std::shared_ptr<CentreData> Ptr;

	virtual void normalise(Matrix& Y);
	virtual void normalise(std::vector<double>& Y);
	virtual void unnormalise(Matrix& Y);
	virtual void unnormalise(std::vector<double>& Y);
};


class Visitable
{
public:
	virtual ~Visitable() {}
	typedef std::shared_ptr<Visitable> Ptr;

	virtual int acceptTrain(TrainVisitor& visitor) = 0;
	virtual std::vector<double> acceptReconstruct(
		ReconstructVisitor& visitor,
		const std::vector<double>& values,
		const std::vector<Constraint::Ptr>& constraints
		) = 0;
};


class LatentModel : public Visitable, public std::enable_shared_from_this<LatentModel>
{
public:
	LatentModel(int numOfLatentDims = 3) : numOfLatentDims(numOfLatentDims), maxY(0) {}
	LatentModel(const Matrix& Y, int numOfLatentDims = 3) : Y(Y), numOfLatentDims(numOfLatentDims), maxY(0) {}

	// Temp until we figure out a way to pass in a matrix directly (bindings)
	void setData(const std::vector<double> &data, int numSamples)
	{
		vectorToMatrix(Y, data, numSamples);
	}

	virtual ~LatentModel() {}
	typedef std::shared_ptr<LatentModel> Ptr;

	void setKernel(Kernel::Ptr kernel)
	{
		this->kernel = kernel;
	}

	void setInitialiseStrategy(InitialiseStrategy::Ptr strategy)
	{
		initialiseStrategy = strategy;
	}

	void setNormaliseStrategy(NormaliseStrategy::Ptr strategy)
	{
		normaliseStrategy = strategy;
	}

	NormaliseStrategy::Ptr getNormaliseStrategy() const
	{
		return normaliseStrategy;
	}

	int getNumOfLatentDims() const { return numOfLatentDims; }
	Kernel::Ptr getKernel() const { return kernel; }

	Matrix& getX() { return X; }
	Matrix& getY() { return Y; }
	Vector& getMeanY() { return meanY; }
	double getMaxY() { return maxY; }

	void setX(const Matrix& X) { this->X = X; }
	void setY(const Matrix& X) { this->Y = Y; }
	void setMeanY(const Vector& meanY) { this->meanY = meanY; }
	void setMaxY(double& maxY) { this->maxY = maxY; }

	virtual int acceptTrain(TrainVisitor& visitor);

	virtual std::vector<double> acceptReconstruct(
		ReconstructVisitor& visitor,
		const std::vector<double>& values,
		const std::vector<Constraint::Ptr>& constraints
		);

public:
	ML_API int save(const std::string& filename);
	ML_API int load(const std::string& filename);

private:
	void readParams(std::vector<Matrix>& parSet, std::ifstream& f_in);
	void writeParams(std::vector<Matrix>& parSet, std::ofstream& f_in);

protected:
	int numOfLatentDims;
	Matrix Y;
	Matrix X;
	Vector meanY;
	double maxY;

	Kernel::Ptr kernel;
	InitialiseStrategy::Ptr initialiseStrategy;
	NormaliseStrategy::Ptr normaliseStrategy;

private:
	// Temp
	void vectorToMatrix(Matrix& Y, const std::vector<double>& data, const int numSamples)
	{
		int nCols = data.size() / numSamples;
		Y = Matrix::Zero(numSamples, nCols);
		for (int r = 0; r < numSamples; r++)
		{
			for (int c = 0; c < nCols; c++)
				Y(r, c) = data[r * nCols + c];
		}
	}
};


class NlOpt : public OptimiseStrategy
{
public:
	NlOpt(
		std::string algorithm = "lbfgs", int maxTrainTime = 60,
		int maxIterations = 1000
		) :
		algorithm(algorithm), maxTrainTime(maxTrainTime),
		maxIterations(maxIterations)
	{}
	typedef std::shared_ptr<NlOpt> Ptr;

	virtual int train(std::vector<double>& data, TrainVisitor* train);
	virtual int reconstruct(
		std::vector<double>& data,
		ReconstructVisitor* reconstruct,
		const std::vector<Constraint::Ptr>& constraints
		);

	std::string getAlgorithm() const { return algorithm; }

private:
	std::string algorithm;
	int maxTrainTime;
	int maxIterations;
};


class Alglib : public OptimiseStrategy
{
public:
	Alglib(
		int maxIterations = 1000, double epsg = 0.01, double epsf = 0,
		double epsx = 0, double stpmax = 0.1
		) :
		maxIterations(maxIterations), epsg(epsg), epsf(epsf),
		epsx(epsx), stpmax(stpmax)
	{}
	typedef std::shared_ptr<Alglib> Ptr;

	virtual int train(std::vector<double>& data, TrainVisitor* train);
	virtual int reconstruct(
		std::vector<double>& data,
		ReconstructVisitor* reconstruct,
		const std::vector<Constraint::Ptr>& constraints
		) {}

	int getMaxIterations() const { return maxIterations; }

private:
	int maxIterations;
	double epsg = 0.01;	// TODO: Use meaningful names
	double epsf = 0;
	double epsx = 0;
	double stpmax = 0.1;
};

}

#endif /* C_QUADRUPEDS_ML_INCLUDE_MODEL_H_ */
