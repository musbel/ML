#include "Model.h"

#include <nlopt.hpp>
#include <libalglib/optimization.h>

#include <boost/algorithm/string.hpp>


namespace ML
{

// -- nlopt --
double estimate_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *func_data)
{
	TrainVisitor* visitor = reinterpret_cast<TrainVisitor*>(func_data);
	return visitor->likelihood(x, grad);
}

double synthesiseConstraints_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *func_data)
{
	Constraint* constraint = reinterpret_cast<Constraint*>(func_data);
	return constraint->synthesise(x, grad);
}

double synthesiseLikelihood_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *func_data)
{
	ReconstructVisitor* visitor = reinterpret_cast<ReconstructVisitor*>(func_data);
	return visitor->likelihood(x, grad);
}

void printNlOptResult(int result)
{
	if (result == NLOPT_FAILURE)
		std::cout << "NLOpt: Generic failure code. " << std::endl;
	else if (result == NLOPT_INVALID_ARGS)
		std::cout << "NLOpt: Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera)." << std::endl;
	else if (result == NLOPT_OUT_OF_MEMORY)
		std::cout << "NLOpt: Ran out of memory." << std::endl;
	else if (result == NLOPT_ROUNDOFF_LIMITED)
		std::cout << "Halted because roundoff errors limited progress. (In this case, the optimisation still typically returns a useful result.)." << std::endl;
	else if (result == NLOPT_FORCED_STOP)
		std::cout << "Halted because of a forced termination: the user called nlopt_force_stop(opt) on the optimization’s nlopt_opt object opt from the user’s objective function or constraints." << std::endl;
	else if (result == NLOPT_SUCCESS)
		std::cout << "Success" << std::endl;
	else if (result == NLOPT_FTOL_REACHED)
		std::cout << "Optimisation stopped because stopval (above) was reached." << std::endl;
	else if (result == NLOPT_FTOL_REACHED)
		std::cout << "Optimisation stopped because ftol_rel or ftol_abs (above) was reached." << std::endl;
	else if (result == NLOPT_XTOL_REACHED)
		std::cout << "Optimisation stopped because xtol_rel or xtol_abs (above) was reached." << std::endl;
	else if (result == NLOPT_MAXEVAL_REACHED)
		std::cout << "Optimisation stopped because maxeval (above) was reached." << std::endl;
	else if (result == NLOPT_MAXTIME_REACHED)
		std::cout << "Optimisation stopped because maxtime (above) was reached." << std::endl;
	else
		std::cout << "This return value is not recognised: " << result << std::endl;
}

int getNlOptAlgorithmKey(const std::string &name)
{
	std::string keyOpt = name;
	boost::algorithm::to_lower(keyOpt);

	if (keyOpt == "lbfgs") return nlopt::LD_LBFGS;
	else if (keyOpt == "slsqp") return nlopt::LD_SLSQP;
	else if (keyOpt == "mma") return nlopt::LD_MMA;
	else if (keyOpt == "lbfgs_nocedal") return nlopt::LD_LBFGS_NOCEDAL;
	else if (keyOpt == "bobyqa") return nlopt::LN_BOBYQA;
	else if (keyOpt == "var1") return nlopt::LD_VAR1;
	else if (keyOpt == "var2") return nlopt::LD_VAR2;
	else if (keyOpt == "newton") return nlopt::LD_TNEWTON_RESTART;
	else if (keyOpt == "newton_precon") return nlopt::LD_TNEWTON_PRECOND_RESTART;
	return -1;
}

int NlOpt::train(std::vector<double>& data, TrainVisitor* train)
{
	if (!train) return 0;

	int optCode = getNlOptAlgorithmKey(algorithm);
	nlopt::opt opt(nlopt::algorithm(optCode), data.size());
	opt.set_vector_storage(40);

	double minf = 0.0;
	opt.set_initial_step(0.000001);	// TODO: Expose parameters
	opt.set_xtol_rel(1e-8);

	if (maxTrainTime != 0) opt.set_maxtime(maxTrainTime);
	if (maxIterations != 0) opt.set_maxeval(maxIterations);

	opt.set_min_objective(estimate_nlopt, train);

	try
	{
		std::cout << "[NlOpt] Optimise (" << algorithm << ")" << std::endl;
		nlopt::result result = opt.optimize(data, minf);

		printNlOptResult(result);
	}
	catch (std::exception& e)
	{
		std::cout << "[Train] Could not finish nlopt optimisation: " << e.what() << std::endl;
		return 0;
	}

	return 1;
}

int NlOpt::reconstruct(
	std::vector<double>& data,
	ReconstructVisitor* reconstruct,
	const std::vector<Constraint::Ptr>& constraints
	)
{
	if (!reconstruct) return 0;

	int optCode = getNlOptAlgorithmKey(algorithm);

	if (optCode == -1)
	{
		std::cout << "Optimisation method is not recognised: " << optCode << std::endl;
		return 0;
	}

	nlopt::opt opt(nlopt::algorithm(optCode), data.size());

	double minf = 0.0;
	opt.set_initial_step(0.000001);	// TODO: Expose parameters
//	opt.set_xtol_rel(1e-8);

	if (maxTrainTime != 0) opt.set_maxtime(maxTrainTime);
	if (maxIterations != 0) opt.set_maxeval(maxIterations);

	opt.set_min_objective(synthesiseLikelihood_nlopt, reconstruct);

	if (constraints.size() && (optCode == nlopt::LD_SLSQP || optCode == nlopt::LD_MMA))
	{
		for (int ci = 0; ci < constraints.size(); ++ci)
		{
			opt.add_inequality_constraint(synthesiseConstraints_nlopt, constraints[ci].get(), 1e-14);
		}
	}

	// TODO: Add lower and upper bounds

	try
	{
		std::cout << "[NlOpt] Optimise (" << algorithm << ")" << std::endl;
		nlopt::result result = opt.optimize(data, minf);

		printNlOptResult(result);
	}
	catch (std::exception& e)
	{
		std::cout << "[Reconstruct] Could not finish nlopt optimisation: " << e.what() << std::endl;
		reconstruct->print();
		return 0;
	}

	return 1;
}

// -- alglib --
void estimate_alglib(const alglib::real_1d_array &x, double &result, alglib::real_1d_array &grad, void *func_data)
{
//	std::cout << "> estimate_alglib" << std::endl;
	TrainVisitor* visitor = reinterpret_cast<TrainVisitor*>(func_data);
	std::vector<double> vecX = std::vector<double>(x.length(), 0.0);
	std::vector<double> vecGrad = std::vector<double>(grad.length(), 0.0);

	for (int i = 0; i < vecX.size(); i++)
		vecX[i] = x[i];

	result = visitor->likelihood(vecX, vecGrad);

	for (int i = 0; i < vecGrad.size(); i++)
		grad[i] = vecGrad[i];
}

int Alglib::train(std::vector<double>& data, TrainVisitor* train)
{
	if (!train) return 0;

	alglib::real_1d_array alglib_X;
	alglib_X.setcontent(data.size(), data.data());
	alglib::ae_int_t maxits = maxIterations;
	alglib::minlbfgsstate state;
	alglib::minlbfgsreport rep;
	alglib::minlbfgscreate(1, alglib_X, state);
	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
	alglib::minlbfgssetstpmax(state, stpmax);
	alglib::minlbfgsoptimize(state, estimate_alglib, NULL, train);
	alglib::minlbfgsresults(state, alglib_X, rep);

	return 0;
}

}
