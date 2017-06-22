
#ifndef C_QUADRUPEDS_ML_INCLUDE_CONSTRAINTS_H_
#define C_QUADRUPEDS_ML_INCLUDE_CONSTRAINTS_H_

#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>

#include "EigenHelper.h"

typedef std::unordered_map<int, int> IntMap;


namespace ML
{

class Constraint
{
public:
	Constraint() : offset(0) {}
	virtual ~Constraint() {}
	typedef std::shared_ptr<Constraint> Ptr;

	virtual double synthesise(const std::vector<double>& x, std::vector<double>& grad) = 0;

	// TODO: Add a function that lets the constraints print or provide some info

	void setOffset(int offset) { this->offset = offset; }
	int getOffset() const { return offset; }

protected:
	int offset;
};

class FeatureConstraint : public Constraint
{
public:
	FeatureConstraint() {}
	FeatureConstraint(int index, double value) : index(index), value(value) {}
	typedef std::shared_ptr<FeatureConstraint> Ptr;

	virtual double synthesise(const std::vector<double>& x, std::vector<double>& grad);

	void setIndex(int i) { index = i; }
	void setValue(double v) { value = v; }

	int getIndex() const { return index; }
	double getValue() const { return value; }

private:
	int index;
	double value;
};

class BoneLengthConstraint : public Constraint
{
public:
	BoneLengthConstraint() : value(0.0) {}

	BoneLengthConstraint(
		const std::vector<int>& set1,
		const std::vector<int>& set2,
		const double value
	)
	: set1(set1), set2(set2), value(value)
	{}

	typedef std::shared_ptr<BoneLengthConstraint> Ptr;

	virtual double synthesise(const std::vector<double>& x, std::vector<double>& grad);

	void addSet(int s1, int s2)
	{
		set1.push_back(s1);
		set2.push_back(s2);
	}

	void setValue(double v) { value = v; }
	double getValue() const { return value; }

private:
	std::vector<int> set1;	// TODO: Shouldn't this be a map?
	std::vector<int> set2;
	double value;
};

}

#endif /* C_QUADRUPEDS_ML_INCLUDE_CONSTRAINTS_H_ */
