/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "pca.h"

void PCA::load_data(const char* data, char sep)
{
	// Read data
	unsigned int row = 0;
	std::ifstream file(data);
	if (file.is_open())
	{
		std::string line, token;
		while (getline(file, line))
		{
			std::stringstream tmp(line);
			unsigned int col = 0;
			while (getline(tmp, token, sep))
			{
				if (X.rows() < row + 1)
				{
					X.conservativeResize(row + 1, X.cols());
				}
				if (X.cols() < col + 1)
				{
					X.conservativeResize(X.rows(), col + 1);
				}
				X(row, col) = atof(token.c_str());
				col++;
			}
			row++;
		}
		file.close();
		Xcentered.resize(X.rows(), X.cols());
	}
	else
		std::cout << "Failed to read file " << data << std::endl;

}

double PCA::kernel(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
	/*
	 Kernels
	 1 = RBF
	 2 = Polynomial
	 TODO - add some of these these:
	 http://crsouza.blogspot.co.uk/2010/03/kernel-functions-for-machine-learning.html
	 */
	switch (kernel_type)
	{
	case 2:
		return (std::pow(a.dot(b) + constant, order));
	default:
		return (std::exp(-gamma * ((a - b).squaredNorm())));
	}

}

void PCA::run_kpca()
{

	// Fill kernel matrix
	K.resize(X.rows(), X.rows());
	for (unsigned int i = 0; i < X.rows(); i++)
	{
		for (unsigned int j = i; j < X.rows(); j++)
		{
			K(i, j) = K(j, i) = kernel(X.row(i), X.row(j));
			//printf("k(%i,%i) = %f\n",i,j,K(i,j));
		}
	}
	//cout << endl << K << endl;

	Eigen::EigenSolver<Eigen::MatrixXd> edecomp(K);
	eigenvalues = edecomp.eigenvalues().real();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	std::vector<std::pair<double, Eigen::VectorXd> > eigen_pairs;
	double c = 0.0;
	for (unsigned int i = 0; i < eigenvectors.cols(); i++)
	{
		if (normalise)
		{
			double norm = eigenvectors.col(i).norm();
			eigenvectors.col(i) /= norm;
		}
		eigen_pairs.push_back(std::make_pair(eigenvalues(i), eigenvectors.col(i)));
	}
	// http://stackoverflow.com/questions/5122804/sorting-with-lambda
	std::sort(eigen_pairs.begin(), eigen_pairs.end(),
			[](const std::pair<double, Eigen::VectorXd> a, const std::pair<double, Eigen::VectorXd> b) -> bool
			{
				return (a.first > b.first);
			});
	for (unsigned int i = 0; i < eigen_pairs.size(); i++)
	{
		eigenvalues(i) = eigen_pairs[i].first;
		c += eigenvalues(i);
		cumulative(i) = c;
		eigenvectors.col(i) = eigen_pairs[i].second;
	}
	transformed.resize(X.rows(), components);

	for (unsigned int i = 0; i < X.rows(); i++)
	{
		for (unsigned int j = 0; j < components; j++)
		{
			for (int k = 0; k < K.rows(); k++)
			{
				transformed(i, j) += K(i, k) * eigenvectors(k, j);
			}
		}
	}

	/*
	 cout << "Input data:" << endl << X << endl << endl;
	 cout << "Centered data:"<< endl << Xcentered << endl << endl;
	 cout << "Centered kernel matrix:" << endl << Kcentered << endl << endl;
	 cout << "Eigenvalues:" << endl << eigenvalues << endl << endl;
	 cout << "Eigenvectors:" << endl << eigenvectors << endl << endl;
	 */
	if (printResults)
	{
		std::cout << "Sorted eigenvalues:" << std::endl;
		for (unsigned int i = 0; i < eigenvalues.rows(); i++)
		{
			if (eigenvalues(i) > 0)
			{
				std::cout << "PC " << i + 1 << ": Eigenvalue: " << eigenvalues(i);
				printf("\t(%3.3f of variance, cumulative =  %3.3f)\n", eigenvalues(i) / eigenvalues.sum(),
						cumulative(i) / eigenvalues.sum());
			}
		}
		std::cout << std::endl;
	}
	//cout << "Sorted eigenvectors:" << endl << eigenvectors << endl << endl;
	//cout << "Transformed data:" << endl << transformed << endl << endl;
}

void PCA::run_pca()
{
	// http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
//	Xcentered = X.rowwise() - X.colwise().mean();
	C = (X.adjoint() * X);// / double(X.rows() - 1.0);
	//Eigen::EigenSolver<Eigen::MatrixXd> edecomp(C, true);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> edecomp(C);

	//eigenvalues = edecomp.eigenvalues();
	eigenvalues = edecomp.eigenvalues().real();
	//eigenvectors = edecomp.eigenvectors();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	std::vector<std::pair<double, Eigen::VectorXd> > eigen_pairs;
	double c = 0.0;
	for (unsigned int i = 0; i < eigenvectors.cols(); i++)
	{
		if (normalise)
		{
			double norm = eigenvectors.col(i).norm();
			eigenvectors.col(i) /= norm;
		}
		eigen_pairs.push_back(std::make_pair(eigenvalues(i), eigenvectors.col(i)));
	}
	// http://stackoverflow.com/questions/5122804/sorting-with-lambda
	std::sort(eigen_pairs.begin(), eigen_pairs.end(),
			[](const std::pair<double, Eigen::VectorXd> a, const std::pair<double, Eigen::VectorXd> b) -> bool
			{
				return (a.first > b.first);
			});
	for (unsigned int i = 0; i < eigen_pairs.size(); i++)
	{
		eigenvalues(i) = eigen_pairs[i].first;
		c += eigenvalues(i);
		cumulative(i) = c;
		eigenvectors.col(i) = eigen_pairs[i].second;
	}
	transformed = X * eigenvectors;
}

void PCA::print()
{
	std::cout << "Input_data:" << std::endl << X << std::endl << std::endl;
//	std::cout << "Centered_data:" << std::endl << Xcentered << std::endl << std::endl;
	std::cout << "Covariance_matrix:" << std::endl << C << std::endl << std::endl;
	std::cout << "Sorted eigenvalues:" << std::endl;
	for (unsigned int i = 0; i < eigenvalues.rows(); i++)
	{
		if (eigenvalues(i) > 0)
		{
			std::cout << "PC " << i + 1 << ": Eigenvalue: " << eigenvalues(i);
			printf("\t(%3.3f of variance, cumulative =  %3.3f)\n", eigenvalues(i) / eigenvalues.sum(), cumulative(i) / eigenvalues.sum());
		}
	}
	std::cout << std::endl;
	std::cout << "Sorted_eigenvectors:" << std::endl << eigenvectors << std::endl << std::endl;
	std::cout << "Transformed_data:" << std::endl << X * eigenvectors << std::endl << std::endl;
	//std::cout << "Transformed centred data:" << std::endl << transformed << std::endl << std::endl;

}

void PCA::write_transformed(std::string file)
{

	std::ofstream outfile(file);
	for (unsigned int i = 0; i < transformed.rows(); i++)
	{
		for (unsigned int j = 0; j < transformed.cols(); j++)
		{
			outfile << transformed(i, j);
			if (j != transformed.cols() - 1) outfile << ",";
		}
		outfile << std::endl;
	}
	outfile.close();
	std::cout << "Written file " << file << std::endl;

}

void PCA::write_eigenvectors(std::string file)
{
	std::ofstream outfile(file);
	for (unsigned int i = 0; i < eigenvectors.rows(); i++)
	{
		for (unsigned int j = 0; j < eigenvectors.cols(); j++)
		{
			outfile << eigenvectors(i, j);
			if (j != eigenvectors.cols() - 1) outfile << ",";
		}
		outfile << std::endl;
	}
	outfile.close();
	std::cout << "Written file " << file << std::endl;
}

/*
 //Example
 * int main(int argc, const char* argv[]){


 if(argc < 2){
 cout << "Usage:\n" << argv[0] << " <DATA>" << endl;
 cout << "File format:\nX1,X2, ... Xn\n";
 return(0);
 }


 PCA* P = new PCA();
 P->load_data("data/test.data");
 P->run_pca();
 std::cout << "Regular PCA (data/test.data):" << std::endl;
 P->run_pca();
 P->print();
 delete P;

 P = new PCA();
 P->load_data("data/wikipedia.data");
 cout << "Kernel PCA (data/wikipedia.data) - RBF kernel, gamma = 0.001:" << endl;
 P->run_kpca();
 P->write_eigenvectors("data/eigenvectors_RBF_data.csv");
 P->write_transformed("data/transformed_RBF_data.csv");
 std::cout << std::endl;
 delete P;

 P = new PCA();
 P->load_data("data/wikipedia.data");
 P->set_kernel(2);
 P->set_constant(1);
 P->set_order(2);
 std::cout << "Kernel PCA (data/wikipedia.data) - Polynomial kernel, order = 2, constant = 1:" << std::endl;
 P->run_kpca();
 P->write_eigenvectors("data/eigenvectors_Polynomial_data.csv");
 P->write_transformed("data/transformed_Polynomial_data.csv");
 std::cout << std::endl;
 delete P;

 return(0);

 }

 */
