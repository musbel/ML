#include "Model.h"


namespace ML
{

int LatentModel::acceptTrain(TrainVisitor& visitor)
{
	if (!initialiseStrategy)
	{
		// TODO: LOG
		std::cout << "No initialise strategy defined." << std::endl;
		return 0;
	}

	if (normaliseStrategy)
		normaliseStrategy->normalise(Y);

	initialiseStrategy->initialise(numOfLatentDims, Y, X, meanY, maxY);
	return visitor.train(shared_from_this(), Y, X);
}

std::vector<double> LatentModel::acceptReconstruct(
	ReconstructVisitor& visitor,
	const std::vector<double>& values,
	const std::vector<Constraint::Ptr>& constraints
	)
{
	std::vector<double> reconstruction = visitor.reconstruct(shared_from_this(), values, constraints);

	if (normaliseStrategy)
		normaliseStrategy->unnormalise(reconstruction);

	return reconstruction;
}


// ------------- REFACTOR (JSON) (and should it be here?) -------------
std::vector<std::string> split(std::string &str, const std::string &delimiters)
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

void LatentModel::writeParams(std::vector<Matrix> &parameters, std::ofstream& f_out)
{
	f_out << "NumPar " << parameters.size() << std::endl;
	for (int i = 0; i < parameters.size(); i++)
	{
		f_out << "Dims " << parameters[i].rows() << " " << parameters[i].cols() << std::endl;
		f_out << parameters[i] << std::endl;
	}
}

void LatentModel::readParams(std::vector<Matrix> &parSet, std::ifstream &f_in)
{
	std::string line;
	std::getline(f_in, line);
	std::vector<std::string> numStrs = split(line, " ");
	int numMat = std::stoi(numStrs[1]);
	parSet = std::vector<Matrix>(numMat, Matrix::Zero(1, 1));

	for (int m = 0; m < numMat; m++)
	{
		std::getline(f_in, line);
		if (line.compare(" ") == 0) continue;
		std::vector<std::string> numStrs = split(line, " ");
		std::vector<int> dims;
		std::transform(numStrs.begin() + 1, numStrs.end(), std::back_inserter(dims), [](const std::string &val)
		{	return std::stoi(val);});

		parSet[m] = Matrix::Zero(dims[0], dims[1]);
		for (int r = 0; r < parSet[m].rows(); r++)
		{
			std::getline(f_in, line);
			std::vector<double> vVals;
			std::vector<cString> numStrs = split(line, " ");
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

int LatentModel::load(const std::string &filename)
{
	std::cout << "Loading latent model: " << filename << std::endl;
	std::ifstream f_in(filename);
	if (!f_in.is_open())
	{
		std::cout << "Could not open file: " << filename << std::endl;
		return 0;
	}

	std::vector<Matrix> parameters;

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
			std::vector<std::string> numStrs = split(line, " ");
			vVals.clear();
			std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
			{	return std::stod(str);});
			meanY = Vector::Zero(vVals.size());
			meanY = Eigen::Map<Vector>(vVals.data(), vVals.size());
		}
		else if (line.compare("Latent Parameters") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> numStrs = split(line, " ");
			int rows = std::stoi(numStrs[0]);
			int cols = std::stoi(numStrs[1]);
			X = Matrix::Zero(rows, cols);
			for (int r = 0; r < rows; r++)
			{
				std::getline(f_in, line);
				numStrs = split(line, " ");
				vVals.clear();
				std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
				{	return std::stod(str);});
				X.row(r) = Eigen::Map<Matrix>(vVals.data(), 1, X.cols());
			}
			numOfLatentDims = X.cols();
		}
		else if (line.compare("Normalized_Y") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> numStrs = split(line, " ");
			int rows = std::stoi(numStrs[0]);
			int cols = std::stoi(numStrs[1]);
			Y = Matrix::Zero(rows, cols);
			for (int r = 0; r < rows; r++)
			{
				std::getline(f_in, line);
				numStrs = split(line, " ");
				vVals.clear();
				std::transform(numStrs.begin(), numStrs.end(), std::back_inserter(vVals), [](std::string &str)
				{	return std::stod(str);});
				Y.row(r) = Eigen::Map<Matrix>(vVals.data(), 1, Y.cols());
			}
		}
		else if (line.compare("normMean") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> meanValues = split(line, " ");
			for (int i = 0; i < meanValues.size(); ++i)
			{
				normMean.push_back(std::stod(meanValues[i]));
			}
		}
		else if (line.compare("stdDev") == 0)
		{
			std::getline(f_in, line);
			std::vector<std::string> stdValues = split(line, " ");
			for (int i = 0; i < stdValues.size(); ++i)
			{
				stdDev.push_back(std::stod(stdValues[i]));
			}
		}
	}

	// TODO: Assuming specifics for now
	kernel = std::make_shared<RbfKernel>(numOfLatentDims);
	kernel->setParameters(parameters);

	normaliseStrategy = std::make_shared<CentreData>();
	normaliseStrategy->mean = Eigen::Map<Matrix>(normMean.data(), 1, normMean.size());
	normaliseStrategy->stdDev = Eigen::Map<Matrix>(stdDev.data(), 1, stdDev.size());

	return 1;
}

int LatentModel::save(const std::string& filename)
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

	if (X.size() != 0)
	{
		f_out << "MaxY\n" << maxY << std::endl;
		f_out << "mean\n" << meanY.array().transpose() << std::endl;
		f_out << "Latent Parameters\n";
		f_out << X.rows() << " " << X.cols() << std::endl;
		f_out << X << std::endl;
		f_out << "Normalized_Y\n";
		f_out << Y.rows() << " " << Y.cols() << std::endl;
		f_out << Y << std::endl;
	}

	if (normaliseStrategy)
	{
		f_out << "normMean\n" << normaliseStrategy->mean << std::endl;
		f_out << "stdDev\n" << normaliseStrategy->stdDev << std::endl;
	}

	f_out.close();
	return 1;
}

}

