
#ifndef C_QUADRUPEDS_ML_INCLUDE_GPLVMIMPL_H_
#define C_QUADRUPEDS_ML_INCLUDE_GPLVMIMPL_H_

#include "Model.h"
#include "TrainModel.h"
#include "ReconstructModel.h"


namespace ML
{

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

class ML_API GPLVMImpl
{
public:
	explicit GPLVMImpl(const int numLatentDims = 3) : numLatentDims(numLatentDims)
	{}

	virtual ~GPLVMImpl() {}

	ML_API int save(const std::string& filename);
	ML_API int load(const std::string &filename);

	ML_API void train(
		const std::vector<double> &data,
		int nSamples,
		cString optPackage = std::string("nlopt"),
		cString optMethod = std::string("LBFGS"),
		double gamma_bc = 0.0001,
		int maxTrainTime = 0,
		int maxNumIter = 2500,
		int activeSetSize = 0,
		bool with_weights = false,
		bool usePeriodicK = false
		);

	ML_API std::vector<double> predict(
		const std::vector<std::string>& featureNames,
		const std::vector<double>& featureValues,
		bool useSoftConstraints = false,
		int maxNumIter = 250,
		int maxTrainTime = 250,
		cString optMethod = "slsqp"
		);

	ML_API void setData(
		const std::vector<int>& dofs,
		const std::vector<double>& boneLengths,
		const std::vector<cString>& jointNames,
		const std::vector<int>& jntInd,
		const std::vector<int>& parentIDs
		);

	ML_API void createBoneLengthConstraints(
		const std::vector<int> &set1,
		const std::vector<int> &set2,
		const std::vector<double> &d_PairConstraints,
		const std::vector<int> &subSetSizes
		);

	ML_API void createFeatureConstraints(
		const std::vector<bool>& b_values,
		const std::vector<double>& d_values
		);

	ML_API int getNumOfLatentDims()
	{
		return model->getNumOfLatentDims();
	}

	LatentModel::Ptr getModel() const
	{
		return model;
	}

	NormaliseStrategy::Ptr getNormaliseStrategy() const
	{
		return normalise;
	}

private:
	void readParams(std::vector<Matrix> &parSet, std::ifstream &f_in);
	void writeParams(std::vector<Matrix> &parSet, std::ofstream &f_in);

private:
	void prepareConstraints(
		const std::vector<std::string>& featureNames,
		const std::vector<double>& featureValues
		);

	void setBonePairs(const std::vector<int> &parentList, const std::vector<int> &jntInd, bool withRoot = false)
	{
		bone_pairs.clear();
		for (int i = 0; i < parentList.size(); i++)
			bone_pairs.push_back(std::pair<int, int>(parentList[i], jntInd[i]));
	}

	// Creates mapping between two sets
	void makeSet(
		const std::vector<int> &set1, const std::vector<int> &set2,
		const std::vector<int> &numElems,
		std::vector<std::pair<std::vector<int>, std::vector<int>>>&mergedSet
		)
	{
		mergedSet.clear();
		for(int i = 0; i < numElems.size(); i++)
		{
			std::vector<int> s1; //(set1.begin() + i, set1.begin() + i + numElems[i]);
			std::vector<int> s2; //(set2.begin() + i, set2.begin() + i + numElems[i]);

			for(int j = 0; j < numElems[j]; j++)
			{
				s1.push_back(set1[i] + j);
				s2.push_back(set2[i] + j);
			}

			std::pair<std::vector<int>, std::vector<int>> new_pair = std::make_pair(s1, s2);
			mergedSet.push_back(new_pair);
		}
	}

private:
	int numLatentDims;
	LatentModel::Ptr model;
	Kernel::Ptr kernel;
	OptimiseStrategy::Ptr optimiser;
	CentreData::Ptr normalise;

	ReconstructGplvm::Ptr reconstructVisitor;

	std::vector<int> dofs;

	// Constraints
	std::vector<Constraint::Ptr> constraints;
	std::vector<std::pair<std::vector<int>, std::vector<int>>> constraintPairs;
	std::vector<double> constraintPairValues;
	std::vector<std::pair<int, int>> bone_pairs;
	std::vector<int> jointSet1;
	std::vector<int> jointSet2;
	std::vector<double> boneLengths;
	std::vector<int> fields;
	std::vector<std::string> jointNames;
	std::vector<bool> b_features;
	std::vector<double> d_features;
	std::vector<bool> b_Constraints;
	std::vector<double> d_Constraints;
};

}

#endif /* C_QUADRUPEDS_ML_INCLUDE_GPLVMIMPL_H_ */
