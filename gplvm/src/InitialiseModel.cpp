#include "Model.h"
#include "pca.h"


namespace ML
{

void PcaInitialiser::initialise(int numLatDims, Matrix& Y, Matrix& X, Vector& meanY, double& maxY)
{
	int numSamples = Y.rows();

	// Add noise to any zero variance columns
	std::vector<double> matVar = EigenHelper::varianceMat<Matrix>(Y, true);
	for (int i = 0; i < Y.cols(); i++)
	{
		if (matVar[i] == 0.0)
			Y.col(i) += Eigen::VectorXd::Random(Y.rows()) * 0.001;
	}

	Matrix centered = Y;
	meanY = centered.array().colwise().mean();

	std::cout << "Initialising with PCA" << std::endl;
	//centered = Y.rowwise() - Y.colwise().mean();

	Vector maxVec = centered.array().abs().colwise().maxCoeff();
	maxY = maxVec.array().maxCoeff();
	if (normalise)
	{
		for (int i = 0; i < maxVec.size(); i++)
		{
			if (maxVec[i] == 0.0)
				continue;

			centered.col(i) = centered.col(i) * (1.0 / maxVec[i]);
		}
	}
	else
	{
		double normMult = 1.0;
		centered = centered.array() * normMult;
	}

	Y = centered;

	std::shared_ptr<PCA> pca = std::make_shared<PCA>(Y);
	pca->set_components(numLatDims);
	pca->set_normalise(int(normalise));
	std::cout << " - Running PCA" << std::endl;
	pca->run_pca();
	std::cout << " - Finished running PCA" << std::endl;
	X = pca->get_transformed(numLatDims);

	std::cout << " - GPLVM initialised with PCA" << std::endl;
}

}

