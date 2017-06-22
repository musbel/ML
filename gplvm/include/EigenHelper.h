
#ifndef C_QUADRUPEDS_ML_EIGENHELPER_H_
#define C_QUADRUPEDS_ML_EIGENHELPER_H_

#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <fstream>
#include <vector>
#include <cmath>

#define RAD2DEGREE(x) (((x)*180.0)/M_PI)
#define DEGREE2RAD(x) (((x)*M_PI)/180.0)

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef std::string cString;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SparseMatrix;


namespace EigenHelper
{

template<typename T>
std::vector<double> varianceMat(const T& mat, bool colwise = true)
{
	std::vector<double> result = (colwise == true) ? std::vector<double>(mat.cols(), 0.0) : std::vector<double>(mat.rows(), 0.0);
	std::vector<double> meanVec = (colwise == true) ? std::vector<double>(mat.cols(), 0.0) : std::vector<double>(mat.rows(), 0.0);

	if (colwise)
	{
		double div = 1.0 / double(mat.rows());

		for (int i = 0; i < mat.cols(); i++)
		{
			meanVec[i] = mat.col(i).sum() * div;
			double sum = 0.0;
			for (int j = 0; j < mat.rows(); j++)
			{
				sum += std::pow(mat(j, i) - meanVec[i], 2.0);
			}
			result[i] = sum * div;
		}
	}
	else
	{
		double div = 1.0 / mat.cols();
		for (int i = 0; i < mat.rows(); i++)
		{
			meanVec[i] = mat.row(i).sum() * div;
			double sum = 0.0;
			for (int j = 0; j < mat.cols(); j++)
				sum += std::pow(mat(i, j) - meanVec[i], 2.0);
			result[i] = sum * div;
		}
	}

	return result;
}

template<typename T>
void abs(std::vector<T> &vmat)
{
	for (int i = 0; i < vmat.size(); i++)
		vmat[i] = vmat[i].array().abs();
}

template<typename T>
void exp(std::vector<T> &vmat)
{
	for (int i = 0; i < vmat.size(); i++)
		vmat[i] = vmat[i].array().exp();
}
template<typename T>
void log(std::vector<T> &vmat)
{
	for (int i = 0; i < vmat.size(); i++)
		vmat[i] = vmat[i].array().log();
}

template<typename T>
void exp(const std::vector<T> &vmat, std::vector<T> &vmat2)
{
	vmat2.clear();
	for (int i = 0; i < vmat.size(); i++)
		vmat2.push_back(vmat[i].array().exp());
}

template<typename T>
void exp(const std::vector<T> &vmat, std::vector<T> &vmat2, std::vector<T> &scales)
{
	vmat2.clear();
	for (int i = 0; i < vmat.size(); i++)
		vmat2.push_back((vmat[i].array() * scales[i].array()).exp());
}

template<typename T>
void log(const std::vector<T> &vmat, std::vector<T> &vmat2)
{
	vmat2.clear();
	for (int i = 0; i < vmat.size(); i++)
		vmat2.push_back(vmat[i].array().log());
}

template<typename Derived>
void getMinIndices(const Eigen::MatrixBase<Derived> &b, std::vector<std::pair<int, int>> &indices, int numEls)
{
	std::vector<std::vector<bool>> flags = std::vector<std::vector<bool>>(b.rows(), std::vector<bool>(b.cols(), false));
	indices.clear();
	if (numEls >= b.size())
	{
		std::cout << "numEls is bigger than the size of the object container, getMinIndices!!!!" << std::endl;
		return;
	}

	for (int i = 0; i < numEls; i++)
	{
		std::pair<int, int> new_ind = std::make_pair(-1, -1);
		double minValue = std::numeric_limits<double>::max();
		for (int r = 0; r < b.rows(); r++)
		{
			for (int c = 0; c < b.cols(); c++)
			{
				if (flags[r][c] == true) continue;
				if (minValue > b(r, c))
				{
					minValue = b(r, c);
					new_ind.first = r;
					new_ind.second = c;
				}
			}
		}

		indices.push_back(new_ind);
		flags[new_ind.first][new_ind.second] = true;
		if (numEls == indices.size()) break;
	}
}

template<typename T, typename K>
void setOffDiagonal(T& mat, const K& val)
{
	for (int r = 0; r < mat.rows(); r++)
		for (int c = 0; c < mat.cols(); c++)
			if (r != c) mat(r, c) = val;
}

template<typename T, typename K>
void setOffDiagonal(std::vector<T>& mat, const K& val)
{
	for (int i = 0; i < mat.size(); i++)
		for (int r = 0; r < mat[i].rows(); r++)
			for (int c = 0; c < mat[i].cols(); c++)
				if (r != c) mat[i](r, c) = val;
}

template<typename T>
void normaliseData(const T& data, T& data_norm, T& data_mean, T& data_std)
{
	data_std = T::Zero(1, data.cols());
	data_mean = T::Zero(1, data.cols());

	double N = double(data.rows());
	data_mean = data.colwise().mean();
	data_norm = data.rowwise() - data.colwise().mean(); // TODO: Use mean
	data_std = data_norm.array().square().colwise().sum() / N;
	data_std = data_std.cwiseSqrt();

	// TODO: Vectorise
	for (int i = 0; i < data.cols(); i++)
	{
		if (std::isless(data_std(i), 1e-10))
			data_std(i) = 1.0;

		data_norm.col(i) /= data_std(0, i);
	}
}

} // Namespace EigenHelper

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> &vec_data)
{
	for (int i = 0; i < vec_data.size(); i++)
	{
		std::string strType = typeid(T).name();
		if (strType.find("Eigen") != std::string::npos)
		{
			if (i != vec_data.size() - 1)
				os << vec_data[i] << std::endl;
			else
				os << vec_data[i];
		}
		else
			os << vec_data[i] << " ";
	}
	return os;
}

#endif /* C_QUADRUPEDS_ML_EIGENHELPER_H_ */
