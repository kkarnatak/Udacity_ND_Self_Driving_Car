#include "PID.h"
#include <iostream>
using namespace std;

///////////////////////////////////////////////////////////////////////////////

const bool DEBUG=true;

///////////////////////////////////////////////////////////////////////////////

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

///////////////////////////////////////////////////////////////////////////////

PID::~PID() {}

///////////////////////////////////////////////////////////////////////////////

void PID::Init(double Kp, double Ki, double Kd)
{
	PID::Kp = Kp;
	PID::Ki = Ki;
	PID::Kd = Kd;
	PID::sum_cte = 0;
	PID::prev_cte = 0;
	PID::d_cte = 0;

	best_error = 9999.0;

	// Initialize deltas
	dp_p_ = Kp/10.0;
	dp_i_ = Ki/10.0;
	dp_d_ = Kd/10.0;

	// Initialize the parameter choice
	parameter_type = 0;

	// If increment in the values needed
	is_to_increment = true;
}

///////////////////////////////////////////////////////////////////////////////

void PID::UpdateError(double cte)
{
	PID::d_cte = cte - PID::prev_cte;
	PID::prev_cte = cte;
}

///////////////////////////////////////////////////////////////////////////////

void PID::TotalError(double cte)
{
	PID::sum_cte += cte;
	if (cte < 0.001)
	{
		PID::sum_cte = 0;
	}
}

///////////////////////////////////////////////////////////////////////////////

double PID::ControlSignal(double cte)
{
	return (-Kp * cte - Kd * d_cte - Ki * sum_cte);
}

///////////////////////////////////////////////////////////////////////////////

void PID::twiddle()
{
	// Took reference from the lecture note
	double current_error = sum_cte;

	// Save copy of current coefficients
	double old_Kp = Kp;
	double old_Ki = Ki;
	double old_Kd = Kd;

	// If current best error is too large
	if (best_error > LARGE_VAL)
	{
		best_error = current_error;

		if (DEBUG)
			std::cout << "\n Best error too big \n Best Param Set (Kp,Ki,Kd) is : " << Kp << Ki << Kd;
		// Increment Kp
		Kp += dp_p_;

		// return
		return;
	}

	// If the current error is less than best error
	if (current_error < best_error)
	{
		// Set best error to current value
		best_error = current_error;

		// Adjust one of the three coefficients
		if (parameter_type == KP)
			dp_p_ *= 1.1;
		else if (parameter_type == KI)
			dp_i_ *= 1.1;
		else
			dp_d_ *= 1.1;

		// Cyclically adjust coefficient choice
		parameter_type = (parameter_type + 1) % 3;

		// Value changed, increment true
		is_to_increment = true;

		if (DEBUG)
			cout << "\n Best Param Set (Kp,Ki,Kd) is :" << Kp << Ki << Kd;
	}
	else
	{
		// AS since the Last update was an increment and current error was worse
		// than best error, set increment to false
		if (is_to_increment == true)
			is_to_increment = false;
		else
		{
			// Since we are decrementing now
			// Adjust Values accordingly based on current coefficient choice
			if (parameter_type == KP)
			{
				Kp += dp_p_;
				dp_p_ *= 0.9;
			}
			else if (parameter_type == KI)
			{
				Ki += dp_i_;
				dp_i_ *= .9;
			}
			else
			{
				Kd += dp_d_;
				dp_d_ *= .9;
			}

			// Cyclically adjust coefficient choice
			parameter_type = (parameter_type + 1) % 3;

			// Since increment was false, Set it to true so next attempt is upwards
			is_to_increment = true;
		}
	}
	// In any case, Readjust Coefficients after conditional check based on coefficient choice
	if (parameter_type == KP)
	{
		if (is_to_increment)
			Kp += dp_p_;
		else
			Kp -= 2 * dp_p_;
	}
	else if (parameter_type == KI)
	{
		if (is_to_increment)
			Ki += dp_i_;
		else
			Ki -= 2 * dp_i_;
	}
	else if (parameter_type == KD)
	{
		if (is_to_increment)
			Kd += dp_d_;
		else
			Kd -= 2 * dp_d_;
	}

	if(DEBUG)
	{
		cout << "\nOLD Kp : " << old_Kp << ", " << "New Kp : " << Kp << endl;
		cout << "\nOLD Ki : " << old_Ki << ", " << "New Ki : " << Ki << endl;
		cout << "\nOLD Kd : " << old_Kd << ", " << "New Kd : " << Kd << endl;
	}
}

///////////////////////////////////////////////////////////////////////////////