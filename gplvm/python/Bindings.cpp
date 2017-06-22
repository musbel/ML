#define BOOST_PYTHON_STATIC_LIB

#include <boost/python/enum.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/dict.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <numpy/arrayobject.h>

#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "Model.h"
#include "TrainModel.h"
#include "ReconstructModel.h"

#include "GPLVMImpl.h" // Temp


using namespace boost::python;

// Const containers on the C++ side
typedef std::vector<std::string> s_List;
typedef std::vector<double>      d_List;
typedef std::vector<float>       f_List;
typedef std::vector<int>         i_List;
typedef std::vector<char>        c_List;
typedef std::vector<bool>        b_List;

// General functions for accessing nicely behaved numpy data in C++
// From here http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
template <typename T> struct NpType {};
template <> struct NpType<double>                {enum { type_code = NPY_DOUBLE };};
template <> struct NpType<unsigned char>         {enum { type_code = NPY_UINT8 };};
template <> struct NpType<int32_t>               {enum { type_code = NPY_INT32 };};
template <> struct NpType<uint16_t>              {enum { type_code = NPY_UINT16 };};
template <> struct NpType<float>                 {enum { type_code = NPY_FLOAT };};
template <> struct NpType<std::complex<double> > {enum { type_code = NPY_CDOUBLE };};

template<class T>
class Np1D {
	const PyArrayObject *_object;
public:
	size_t _size;
	T* _data;
	Np1D(object &y, int sz=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np1D: Not a numpy array");
		if (_object->nd != 1) throw std::runtime_error("Np1D: Must be 1D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np1D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np1D: Numpy array is not contiguous");
		if (int(PyArray_TYPE(_object)) != int(NpType<T>::type_code)) {
			//std::cerr << int(PyArray_TYPE(_object)) << "  " << int(NpType<T>::type_code) <<  std::endl;
			if (int(PyArray_TYPE(_object)) == NPY_INT && int(NpType<T>::type_code) == NPY_INT32) {
				//std::cerr << "(W) Warning Np1D: numpy dtype is int, expected int32?" << std::endl;
			}
			else {
				throw std::runtime_error("Np1D: Wrong numpy dtype");
			}
		}
		_size = _object->dimensions[0];
		if (sz != -1 && _size != sz) throw std::runtime_error("Np1D: Wrong size");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np1D() { Py_DECREF(_object); }
	T& operator[](size_t i) { assert(i >= 0 && i < _size); return _data[i]; }
	const T operator[](size_t i) const { assert(i >= 0 && i < _size); return _data[i]; }
	T max(const T &d) const {
		size_t size = _size;
		if (size < 1) return d;
		return *std::max_element(_data,_data+size);
	}
	bool any() { return _size > 0; }
};

template<class T>
class Np2D {
	const PyArrayObject *_object;
public:
	size_t _rows, _cols;
	T* _data;
	Np2D(object &y, int rows=-1, int cols=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np2D: Not a numpy array");
		if (_object->nd != 2) throw std::runtime_error("Np2D: Must be 2D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np2D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np2D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np2D: Wrong numpy dtype");
		_rows = _object->dimensions[0];
		_cols = _object->dimensions[1];
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np2D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np2D: Wrong number of cols");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np2D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _rows); return _data + i*_cols; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _rows); return _data + i*_cols; }
	T max(const T &d) const {
		size_t size = _rows*_cols;
		if (size < 1) return d;
		return *std::max_element(_data,_data+size);
	}
	bool any() { return (_rows != -1 && _cols != -1); }
};

template<class T>
class Np3D {
	const PyArrayObject *_object;
public:
	size_t _rows, _cols, _chans;
	T* _data;
	Np3D(object &y, int rows=-1, int cols=-1, int chans=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np3D: Not a numpy array");
		if (_object->nd != 3) throw std::runtime_error("Np3D: Must be 3D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np3D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np3D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np3D: Wrong numpy dtype");
		_rows = _object->dimensions[0];
		_cols = _object->dimensions[1];
		_chans = _object->dimensions[2];
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np3D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np3D: Wrong number of cols");
		if (chans != -1 && _chans != chans) throw std::runtime_error("Np3D: Wrong number of chans");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np3D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _rows); return _data + i*_cols*_chans; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _rows); return _data + i*_cols*_chans; }
	bool any() { return (_rows != -1 && _cols != -1 && _chans != -1); }
};

template<class T>
class Np4D {
	const PyArrayObject *_object;
public:
	size_t _items, _rows, _cols, _chans;
	T* _data;
	Np4D(object &y, int items=-1,int rows=-1, int cols=-1, int chans=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np4D: Not a numpy array");
		if (_object->nd != 4) throw std::runtime_error("Np4D: Must be 4D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np4D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np4D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np4D: Wrong numpy dtype");
		_items = _object->dimensions[0];
		_rows = _object->dimensions[1];
		_cols = _object->dimensions[2];
		_chans = _object->dimensions[3];
		if (items != -1 && _items != items) throw std::runtime_error("Np4D: Wrong number of items");
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np4D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np4D: Wrong number of cols");
		if (chans != -1 && _chans != chans) throw std::runtime_error("Np4D: Wrong number of chans");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np4D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _items); return _data + i*_rows*_cols*_chans; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _items); return _data + i*_rows*_cols*_chans; }
};


template<class T>
object newArray(const int size)
{
	npy_intp py_size = size;
	PyObject *pobj = PyArray_SimpleNew(1, &py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArray2D(const int rows, const int cols)
{
	npy_intp py_size[2] = { rows, cols };
	PyObject *pobj = PyArray_SimpleNew(2, py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArray3D(const int rows, const int cols, const int chans)
{
	npy_intp py_size[3] = { rows, cols, chans };
	PyObject *pobj = PyArray_SimpleNew(3, py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArrayFromVector(const std::vector<T> &v)
{
	const int size = v.size();
	object y(newArray<T>(size));
	if (size) {
		Np1D<T> o(y);
		std::copy(v.begin(), v.end(), o._data);
	}
	return y;
}

template<class T>
object newArray2DFromVector(const std::vector<T> &v, const int stride)
{
	const int size = v.size() / stride;
	object y(newArray2D<T>(size, stride));
	if (size) {
		Np2D<T> o(y);
		std::copy(v.begin(), v.end(), o._data);
	}
	return y;
}

struct iterable_converter
{
	/// @note Registers converter from a Python iterable type to the
	///       provided type.
	template<typename Container>
	iterable_converter&
	from_python()
	{
		boost::python::converter::registry::push_back(&iterable_converter::convertible, &iterable_converter::construct<Container>,
				boost::python::type_id<Container>());

		// Support chaining.
		return *this;
	}

	/// @brief Check if PyObject is iterable.
	static void* convertible(PyObject* object)
	{
		return PyObject_GetIter(object) ? object : NULL;
	}

	/// @brief Convert iterable PyObject to C++ container type.
	///
	/// Container Concept requirements:
	///
	///   * Container::value_type is CopyConstructable.
	///   * Container can be constructed and populated with two iterators.
	///     I.e. Container(begin, end)
	template<typename Container>
	static void construct(PyObject* object, boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		namespace python = boost::python;
		// Object is a borrowed reference, so create a handle indicating it is
		// borrowed for proper reference counting.
		python::handle<> handle(python::borrowed(object));

		// Obtain a handle to the memory block that the converter has allocated
		// for the C++ type.
		typedef python::converter::rvalue_from_python_storage<Container> storage_type;
		void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

		typedef python::stl_input_iterator<typename Container::value_type> iterator;

		// Allocate the C++ type into the converter's memory block, and assign
		// its handle to the converter's convertible variable.  The C++
		// container is populated by passing the begin and end iterators of
		// the python object to the container's constructor.

		new (storage) Container(iterator(python::object(handle)), iterator());
		data->convertible = storage;
	}
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gplvm_train_bc_overloads, train, 2, 10)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(predict_overloads, predict, 2, 6)

namespace ML_PYTHON
{
	std::vector<double> CentreData_normalise(ML::LatentModel::Ptr model, const std::vector<double>& Y)
	{
		ML::NormaliseStrategy::Ptr normaliser = model->getNormaliseStrategy();
		if (normaliser)
		{
			std::vector<double> temp = Y;
			normaliser->normalise(temp);
			return temp;
		}

		return Y;
	}

	std::vector<double> CentreData_unnormalise(ML::LatentModel::Ptr model, const std::vector<double>& Y)
	{
		ML::NormaliseStrategy::Ptr normaliser = model->getNormaliseStrategy();
		if (normaliser)
		{
			std::vector<double> temp = Y;
			normaliser->unnormalise(temp);
			return temp;
		}

		return Y;
	}
}

BOOST_PYTHON_MODULE(ML)
{
	import_array();
	numeric::array::set_module_and_type("numpy", "ndarray");

	class_<s_List>("str_List")
		.def(vector_indexing_suite<s_List>());
	class_<b_List>("b_List")
		.def(vector_indexing_suite<b_List>());
	class_<c_List>("c_List")
		.def(vector_indexing_suite<c_List>());
	class_<i_List>("i_List")
		.def(vector_indexing_suite<i_List>());
	class_<f_List>("f_List")
		.def(vector_indexing_suite<f_List>());
	class_<d_List>("d_List")
		.def(vector_indexing_suite<d_List>());

	// Register iterable conversions.
	iterable_converter()
		// Built-in type.
		.from_python<std::vector<double>>()
		.from_python<std::vector<float>>()
		.from_python<std::vector<int>>()
		.from_python<std::vector<char>>()
		.from_python<std::vector<bool>>()
		.from_python<std::vector<std::string>>()
		.from_python<std::vector<ML::Constraint::Ptr>>()
		.from_python<std::map<int, int>>()
		// User type.
		.from_python<std::list<bool>>()
		.from_python<std::list<char>>()
		.from_python<std::list<int> >()
		.from_python<std::list<float>>()
		.from_python<std::list<double>>()
		.from_python<std::list<std::string>>();

	class_<ML::LatentModel, ML::LatentModel::Ptr>("LatentModel")
		.def(init<optional<int>>())
		.def("setData", &ML::LatentModel::setData)	// Temp
		// TODO: Initialise with Matrix Y and int
		.def("getNumOfLatentDims", &ML::LatentModel::getNumOfLatentDims)
		.def("setKernel", &ML::LatentModel::setKernel)
		.def("setInitialiseStrategy", &ML::LatentModel::setInitialiseStrategy)
		.def("setNormaliseStrategy", &ML::LatentModel::setNormaliseStrategy)
		.def("acceptTrain", &ML::LatentModel::acceptTrain)
		.def("acceptReconstruct", &ML::LatentModel::acceptReconstruct)
		.def("load", &ML::LatentModel::load)
		.def("save", &ML::LatentModel::save)
		;

	class_<ML::Kernel, boost::noncopyable>("Kernel", no_init)
		.def("getNumParams", &ML::Kernel::getNumParams)
		;

	class_<ML::RbfKernel, ML::RbfKernel::Ptr, bases<ML::Kernel>>("RbfKernel")
		.def(init<optional<int>>())
		;

	class_<ML::OptimiseStrategy, boost::noncopyable>("OptimiseStrategy", no_init);

	class_<ML::NlOpt, ML::NlOpt::Ptr, bases<ML::OptimiseStrategy>>("NlOpt")
		.def(init<optional<std::string, int, int>>())
		.def("getAlgorithm", &ML::NlOpt::getAlgorithm)
		;

	class_<ML::Alglib, ML::Alglib::Ptr, bases<ML::OptimiseStrategy>>("Alglib")
		.def(init<optional<int, double, double, double, double>>())
		.def("getMaxIterations", &ML::Alglib::getMaxIterations)
		;

	class_<ML::InitialiseStrategy, boost::noncopyable>("InitialiseStrategy", no_init);

	class_<ML::PcaInitialiser, ML::PcaInitialiser::Ptr, bases<ML::InitialiseStrategy>>("PcaInitialiser")
		.def(init<optional<bool>>())
		.def("getNumSamples", &ML::PcaInitialiser::getNumSamples)
		;

	class_<ML::TrainVisitor, boost::noncopyable>("TrainVisitor", no_init)
		.def("setKernel", &ML::TrainVisitor::setKernel)
		.def("setOptimiseStrategy", &ML::TrainVisitor::setOptimiseStrategy)
		;

	class_<ML::TrainGplvm, ML::TrainGplvm::Ptr, bases<ML::TrainVisitor>>("TrainGplvm")
		.def(init<>())
		.def("getStepSize", &ML::TrainGplvm::getStepSize)
		;

	class_<ML::TrainGplvmWithBc, ML::TrainGplvmWithBc::Ptr, bases<ML::TrainVisitor>>("TrainGplvmWithBc")
		.def(init<optional<double>>())
		.def("getGamma", &ML::TrainGplvmWithBc::getGamma)
		;

	class_<ML::ReconstructVisitor, boost::noncopyable>("ReconstructVisitor", no_init)
		.def("setKernel", &ML::ReconstructVisitor::setKernel)
		.def("setOptimiseStrategy", &ML::ReconstructVisitor::setOptimiseStrategy)
		;

	class_<ML::ReconstructGplvm, ML::ReconstructGplvm::Ptr, bases<ML::ReconstructVisitor>>("ReconstructGplvm")
		.def(init<>())
		;

	class_<ML::NormaliseStrategy, boost::noncopyable>("NormaliseStrategy", no_init)
		;

	def("normalise", &ML_PYTHON::CentreData_normalise); //.staticmethod("CentreData_normalise")
	def("unnormalise", &ML_PYTHON::CentreData_unnormalise); //.staticmethod("CentreData_normalise")
	class_<ML::CentreData, ML::CentreData::Ptr, bases<ML::NormaliseStrategy>>("CentreData")
		.def(init<>())
		;

	class_<ML::Constraint, boost::noncopyable>("Constraint", no_init);

	class_<ML::FeatureConstraint, ML::FeatureConstraint::Ptr, bases<ML::Constraint>>("FeatureConstraint")
		.def(init<>())
		.def(init<int, double>())
		.def("setIndex", &ML::FeatureConstraint::setIndex)
		.def("setValue", &ML::FeatureConstraint::setValue)
		.def("getIndex", &ML::FeatureConstraint::getIndex)
		.def("getValue", &ML::FeatureConstraint::getValue)
		;

	class_<ML::BoneLengthConstraint, ML::BoneLengthConstraint::Ptr, bases<ML::Constraint>>("BoneLengthConstraint")
		.def(init<>())
		.def(init<std::vector<int>, std::vector<int>, double>())
		.def("addSet", &ML::BoneLengthConstraint::addSet)
		.def("setValue", &ML::BoneLengthConstraint::setValue)
		.def("getValue", &ML::BoneLengthConstraint::getValue)
		;

	class_<ML::GPLVMImpl, std::shared_ptr<ML::GPLVMImpl>>("GPLVM")
		.def(init<optional<int>>())
		.def("train", &ML::GPLVMImpl::train, gplvm_train_bc_overloads(args("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"), "Train GPLVM model"))
		.def("load", &ML::GPLVMImpl::load)
		.def("save", &ML::GPLVMImpl::save)
		.def("getNumOfLatentDims", &ML::GPLVMImpl::getNumOfLatentDims)
		.def("setData", &ML::GPLVMImpl::setData)
		.def("predict", &ML::GPLVMImpl::predict, predict_overloads (args("x1", "x2", "x3", "x4", "x5", "x6", "x7"), "Predict missing data"))
		.def("getModel", &ML::GPLVMImpl::getModel) // Hack alert: To allow combining new models with old monolithic code
//		.def("setNormalisationData", &ML::GPLVMImpl::setNormalisationData)
//		.def("getMean", &ML::GPLVMImpl::getMean)
//		.def("getStd", &ML::GPLVMImpl::getStd)
//		.def("print", &ML::GPLVMImpl::print)
		;

#ifdef WIN32
	register_ptr_to_python<ML::LatentModel::Ptr>();
	register_ptr_to_python<ML::Kernel::Ptr>();
	register_ptr_to_python<ML::RbfKernel::Ptr>();
	register_ptr_to_python<ML::InitialiseStrategy::Ptr>();
	register_ptr_to_python<ML::PcaInitialiser::Ptr>();
	register_ptr_to_python<ML::OptimiseStrategy::Ptr>();
	register_ptr_to_python<ML::NlOpt::Ptr>();
	register_ptr_to_python<ML::Alglib::Ptr>();
	register_ptr_to_python<ML::TrainVisitor::Ptr>();
	register_ptr_to_python<ML::TrainGplvm::Ptr>();
	register_ptr_to_python<ML::TrainGplvmWithBc::Ptr>();
	register_ptr_to_python<ML::ReconstructVisitor::Ptr>();
	register_ptr_to_python<ML::ReconstructGplvm::Ptr>();
	register_ptr_to_python<ML::Constraint::Ptr>();
	register_ptr_to_python<ML::FeatureConstraint::Ptr>();
	register_ptr_to_python<ML::BoneLengthConstraint::Ptr>();
	register_ptr_to_python<ML::NormaliseStrategy::Ptr>();
	register_ptr_to_python<ML::CentreData::Ptr>();
#endif

	implicitly_convertible<ML::RbfKernel::Ptr, ML::Kernel::Ptr>();
	implicitly_convertible<ML::PcaInitialiser::Ptr, ML::InitialiseStrategy::Ptr>();
	implicitly_convertible<ML::NlOpt::Ptr, ML::OptimiseStrategy::Ptr>();
	implicitly_convertible<ML::Alglib::Ptr, ML::OptimiseStrategy::Ptr>();
	implicitly_convertible<ML::TrainGplvm::Ptr, ML::TrainVisitor::Ptr>();
	implicitly_convertible<ML::TrainGplvmWithBc::Ptr, ML::TrainVisitor::Ptr>();
	implicitly_convertible<ML::ReconstructGplvm::Ptr, ML::ReconstructVisitor::Ptr>();
	implicitly_convertible<ML::FeatureConstraint::Ptr, ML::Constraint::Ptr>();
	implicitly_convertible<ML::BoneLengthConstraint::Ptr, ML::Constraint::Ptr>();
	implicitly_convertible<ML::CentreData::Ptr, ML::NormaliseStrategy::Ptr>();

}
