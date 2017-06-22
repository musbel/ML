#ifndef PCA_H
#define PCA_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <utility> 

#include <assert.h>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

class PCA
{
public:

	PCA() : components(2), kernel_type(1), normalise(0), gamma(0.001), constant(1.0), order(2.0)
	{
		printResults = false;
	}

	explicit PCA(Eigen::MatrixXd& d) :
			components(2), kernel_type(1), normalise(0), gamma(0.001), constant(1.0), order(2.0)
	{
		X = d;
		printResults = false;
	}

	void load_data(const char* data, char sep = ',');

	void set_data(Eigen::MatrixXd &matY)
	{
		this->X = matY;
	}

	void set_components(const int i)
	{
		components = i;
	}
	;

	void set_kernel(const int i)
	{
		kernel_type = i;
	}
	;

	void set_normalise(const int i)
	{
		normalise = i;
	}
	;

	void set_gamma(const double i)
	{
		gamma = i;
	}
	;

	void set_constant(const double i)
	{
		constant = i;
	}
	;

	void set_order(const double i)
	{
		order = i;
	}
	;

	Eigen::MatrixXd& get_transformed()
	{
		return transformed;
	}

	Eigen::MatrixXd get_transformed(const int numDims)
	{
		Eigen::MatrixXd kk = this->transformed.block(0, 0, this->X.rows(), numDims);
		return kk;
	}

	Eigen::MatrixXd get_eigenVectors(const int numDim)
	{
		return this->eigenvectors.block(0, 0, this->eigenvectors.rows(), numDim);
	}
	void run_pca();
	void run_kpca();
	void print();
	void write_transformed(std::string);
	void write_eigenvectors(std::string);

private:
	double kernel(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
	Eigen::MatrixXd X, Xcentered, C, K, eigenvectors, transformed;
	Eigen::VectorXd eigenvalues, cumulative;
	unsigned int components, kernel_type, normalise;
	double gamma, constant, order;
	bool printResults;

};

#endif // PCA_H
