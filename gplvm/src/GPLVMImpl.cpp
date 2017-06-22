#include "GPLVMImpl.h"

#include <iterator>


namespace ML
{

void GPLVMImpl::train(
	const std::vector<double> &data,
	int nSamples,
	cString optPackage,
	cString optMethod,
	double gamma_bc,
	int maxTrainTime,
	int maxNumIters,
	int activeSetSize,
	bool with_weights,
	bool usePeriodicK
	)
{
	Matrix Y;
	vectorToMatrix(Y, data, nSamples);

	model = std::make_shared<LatentModel>(Y, numLatentDims);
	kernel = std::make_shared<RbfKernel>(1); // TODO: Implement multi-dimensional

	OptimiseStrategy::Ptr optimiser;
	if (optPackage == "nlopt" )
		optimiser = std::make_shared<NlOpt>(optMethod, maxTrainTime, maxNumIters);
	else if (optPackage == "alglib")
		optimiser = std::make_shared<Alglib>(maxNumIters);

	if (!optimiser)
	{
		std::cout << "No optimisation strategy defined";
		return;
	}

	TrainGplvmWithBc train(gamma_bc);
//	TrainGplvm train();
	train.setOptimiseStrategy(optimiser);
	InitialiseStrategy::Ptr initPca = std::make_shared<PcaInitialiser>();

	model->setInitialiseStrategy(initPca);
	model->acceptTrain(train);
}

std::vector<double> GPLVMImpl::predict(
	const std::vector<std::string>& featureNames,
	const std::vector<double>& featureValues,
	bool useSoftConstraints,
	int maxNumIters,
	int maxTrainTime,
	std::string optMethod
	)
{
	if (!optimiser)
	{
		optimiser = std::make_shared<NlOpt>(optMethod, maxTrainTime, maxNumIters);
	}

	if (!reconstructVisitor)
	{
		reconstructVisitor = std::make_shared<ReconstructGplvm>();
		reconstructVisitor->setOptimiseStrategy(optimiser);
	}

	prepareConstraints(featureNames, featureValues);

	return model->acceptReconstruct(*reconstructVisitor, d_features, constraints);
}

void GPLVMImpl::prepareConstraints(
	const std::vector<std::string>& featureNames,
	const std::vector<double>& featureValues
	)
{
	constraints.clear();

	int nCols = std::accumulate(dofs.begin(), dofs.end(), 0);
	int nSamples = featureValues.size() / (featureNames.size() * 3);

	jointSet1 = bone_pairs[0].first == -1 ? std::vector<int>(bone_pairs.size() - 1, 0) : std::vector<int>(bone_pairs.size(), 0);
	jointSet2 = std::vector<int>(jointSet1.size(), 0);

	int n = 0;
	for (int i = 0; i < bone_pairs.size(); i++)
	{
		if (bone_pairs[i].first == -1) continue;

		jointSet1[n] = bone_pairs[i].first * 3;
		jointSet2[n] = bone_pairs[i].second * 3;
		n++;
	}

	b_features = std::vector<bool>(nCols, false);
	d_features = std::vector<double>(nSamples * nCols, 0.0);

	// Set the given fields
	int f = 0;
	for (int r = 0; r < featureNames.size(); r++)
	{
		int d = 0;
		for (int s = 0; s < jointNames.size(); s++)
		{
			if (featureNames[r].compare(jointNames[s]) == 0)
			{
				for (int m = 0; m < dofs[s]; m++)
				{
					b_features[d + m] = true;
					d_features[d + m] = featureValues[f++];
				}

				break;
			}

			d += dofs[s];
		}
	}

	fields.clear();
	std::copy(dofs.begin(), dofs.end() - 1, std::back_inserter(fields));

	createFeatureConstraints(b_features, d_features);
	createBoneLengthConstraints(jointSet1, jointSet2, boneLengths, fields);
}

void GPLVMImpl::setData(
	const std::vector<int>& dofs,
	const std::vector<double>& boneLengths,
	const std::vector<cString>& jointNames,
	const std::vector<int>& jntInd,
	const std::vector<int>& parentIDs
	)
{
	this->dofs = dofs;
	this->boneLengths = boneLengths;
	this->jointNames = jointNames;

	setBonePairs(parentIDs, jntInd);
}

void GPLVMImpl::createBoneLengthConstraints(
	const std::vector<int> &set1,
	const std::vector<int> &set2,
	const std::vector<double> &boneLengths,
	const std::vector<int> &subSetSizes
	)
{
	int index = 0;
	for (int i = 0; i < subSetSizes.size(); i++)
	{
		BoneLengthConstraint::Ptr constraint = std::make_shared<BoneLengthConstraint>();
		for (int j = 0; j < subSetSizes[i]; j++)
		{
			constraint->addSet(set1[index] + j, set2[index] + j);
		}

		index++;
		constraint->setValue(boneLengths[i]);
		constraints.push_back(constraint);
	}
}

void GPLVMImpl::createFeatureConstraints(
	const std::vector<bool>& b_values,
	const std::vector<double>& d_values
	)
{
	int index = 0;
	for (int i = 0; i < b_values.size(); i++)
	{
		if (b_values[i] == false) // If the features are not given
			continue;

		FeatureConstraint::Ptr constraint = std::make_shared<FeatureConstraint>();
		constraint->setIndex(i);
		constraint->setValue(d_values[i]);
		constraints.push_back(constraint);
	}
}

// ------------- REFACTOR (JSON) -------------
std::vector<std::string> split2(std::string &str, const std::string &delimiters)
{
	char *cstr = new char[str.length() + 1];
	strcpy(cstr, str.c_str());

	char *token = std::strtok(cstr, delimiters.c_str());
	std::vector<std::string> internal;
	std::stringstream ss(str); // Turn the string into a stream.
	std::string tok;

	while (token != NULL)
	{
		internal.push_back(std::string(token));
		token = std::strtok(NULL, delimiters.c_str());
	}

	return internal;
}

void GPLVMImpl::writeParams(std::vector<Matrix> &parameters, std::ofstream& f_out)
{
	f_out << "NumPar " << parameters.size() << std::endl;
	for (int i = 0; i < parameters.size(); i++)
	{
		f_out << "Dims " << parameters[i].rows() << " " << parameters[i].cols() << std::endl;
		f_out << parameters[i] << std::endl;
	}
}

void GPLVMImpl::readParams(std::vector<Matrix> &parSet, std::ifstream &f_in)
{
	std::string line;
	std::getline(f_in, line);
	std::vector<std::string> numStrs = split2(line, " ");
	int numMat = std::stoi(numStrs[1]);
	parSet = std::vector<Matrix>(numMat, Matrix::Zero(1, 1));

	for (int m = 0; m < numMat; m++)
	{
		std::getline(f_in, line);
		if (line.compare(" ") == 0) continue;
		std::vector<std::string> numStrs = split2(line, " ");
		std::vector<int> dims;
		std::transform(numStrs.begin() + 1, numStrs.end(), std::back_inserter(dims), [](const std::string &val)
		{	return std::stoi(val);});

		parSet[m] = Matrix::Zero(dims[0], dims[1]);
		for (int r = 0; r < parSet[m].rows(); r++)
		{
			std::getline(f_in, line);
			std::vector<double> vVals;
			std::vector<cString> numStrs = split2(line, " ");
			std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](const cString &val)
			{
				return std::stod(val);
			});

			for (int c = 0; c < dims[1]; c++)
			{
				parSet[m](r, c) = vVals[c];
			}
		}
	}
}

int GPLVMImpl::load(const std::string &filename)
{
	std::cout << "Loading GPLVM model: " << filename << std::endl;
	std::ifstream f_in(filename);
	if (!f_in.is_open())
	{
		std::cout << "Could not open file: " << filename << std::endl;
		return 0;
	}

	int numLatentDims;
	std::vector<Matrix> parameters;
	Matrix Y;
	Matrix X;
	Vector meanY;
	double maxY;

	std::vector<double> normMean;
	std::vector<double> stdDev;

	std::string line;
	std::vector<double> vVals;
	while (std::getline(f_in, line))
	{
		if (line.compare("NormalParams") == 0)
		{
			readParams(parameters, f_in);
		}
		else if (line.compare("MaxY") == 0)
		{
			std::getline(f_in, line);
			maxY = std::stod(line);
		}
		else if (line.compare("mean") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> numStrs = split2(line, " ");
			vVals.clear();
			std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
			{	return std::stod(str);});
			meanY = Vector::Zero(vVals.size());
			meanY = Eigen::Map<Vector>(vVals.data(), vVals.size());
		}
		else if (line.compare("Latent Parameters") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> numStrs = split2(line, " ");
			int rows = std::stoi(numStrs[0]);
			int cols = std::stoi(numStrs[1]);
			X = Matrix::Zero(rows, cols);
			for (int r = 0; r < rows; r++)
			{
				std::getline(f_in, line);
				numStrs = split2(line, " ");
				vVals.clear();
				std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
				{	return std::stod(str);});
				X.row(r) = Eigen::Map<Matrix>(vVals.data(), 1, X.cols());
			}
			numLatentDims = X.cols();
		}
		else if (line.compare("Normalized_Y") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> numStrs = split2(line, " ");
			int rows = std::stoi(numStrs[0]);
			int cols = std::stoi(numStrs[1]);
			Y = Matrix::Zero(rows, cols);
			for (int r = 0; r < rows; r++)
			{
				std::getline(f_in, line);
				numStrs = split2(line, " ");
				vVals.clear();
				std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
				{	return std::stod(str);});
				Y.row(r) = Eigen::Map<Matrix>(vVals.data(), 1, Y.cols());
			}
		}
		else if (line.compare("normMean") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> meanValues = split2(line, " ");
			for (int i = 0; i < meanValues.size(); ++i)
			{
				normMean.push_back(std::stod(meanValues[i]));
			}
		}
		else if (line.compare("stdDev") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> stdValues = split2(line, " ");
			for (int i = 0; i < stdValues.size(); ++i)
			{
				stdDev.push_back(std::stod(stdValues[i]));
			}
		}
	}

	// TODO: Assuming specifics for now
	model = std::make_shared<LatentModel>(Y, numLatentDims);
	model->setX(X);
	model->setY(Y);
	model->setMeanY(meanY);
	model->setMaxY(maxY);

	kernel = std::make_shared<RbfKernel>(numLatentDims);
	kernel->setParameters(parameters);
	model->setKernel(kernel);

	normalise = std::make_shared<CentreData>();
	normalise->mean = Eigen::Map<Matrix>(normMean.data(), 1, normMean.size());
	normalise->stdDev = Eigen::Map<Matrix>(stdDev.data(), 1, stdDev.size());
	model->setNormaliseStrategy(normalise);

	return 1;
}

int GPLVMImpl::save(const std::string& filename)
{
	std::ofstream f_out(filename);
	if (!f_out.is_open())
	{
		std::cout << "Error trying to save model: Could not open file " << filename << std::endl;
		return 0;
	}

	f_out << "With_BackConstraints\n" << "Back Constraint Model" << std::endl;
	if (kernel)
	{
		std::vector<Matrix> parameters = kernel->getParameters();
		if (parameters.size() != 0)
		{
			f_out << "NormalParams\n";
			writeParams(parameters, f_out);
		}
	}

	if (model)
	{
		Matrix X = model->getX();
		Matrix Y = model->getY();
		if (X.size() != 0)
		{
			f_out << "MaxY\n" << model->getMaxY() << std::endl;
			f_out << "mean\n" << model->getMeanY().array().transpose() << std::endl;
			f_out << "Latent Parameters\n";
			f_out << X.rows() << " " << X.cols() << std::endl;
			f_out << X << std::endl;
			f_out << "Normalized_Y\n";
			f_out << Y.rows() << " " << Y.cols() << std::endl;
			f_out << Y << std::endl;
		}
	}

	if (normalise)
	{
		f_out << "normMean\n" << normalise->mean << std::endl;
		f_out << "stdDev\n" << normalise->stdDev << std::endl;
	}

	f_out.close();
	return 1;
}

}
