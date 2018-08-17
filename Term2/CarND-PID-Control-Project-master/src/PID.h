#ifndef PID_H
#define PID_H

#include <math.h>

///////////////////////////////////////////////////////////////////////////////

#define LARGE_VAL 999
#define KP 0
#define KI 1
#define KD 2

///////////////////////////////////////////////////////////////////////////////

class PID {
public:
  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;
  double sum_cte;
  double prev_cte;
  double d_cte;

	// current value of best error
	double best_error;

	// Deltas for coefficients
	double dp_p_, dp_i_, dp_d_;

	// Variable to determine whether Kp, Ki or Kd are adjusted in a twiddle cycle
	int parameter_type;

	// Variable determining whether to increment or decrement deltas in twiddle
	bool is_to_increment;
  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  void TotalError(double cte);
  double ControlSignal(double cte);

	void twiddle();
};

#endif /* PID_H */
