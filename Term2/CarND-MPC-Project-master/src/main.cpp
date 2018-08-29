#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "MPC.h"
#include "matplotlibcpp.h"

///////////////////////////////////////////////////////////////////////////////

namespace plt = matplotlibcpp;

// for convenience
using json = nlohmann::json;
static int PLOT_COUNTER = 1;

// Set to enable plotting
static bool IS_PLOT_ENABLED = false;

///////////////////////////////////////////////////////////////////////////////

/*
Method inspired by the quiz mpc_to_line. Using same idea to plot cte, velocity
and delta. This will help to visualize the change of these parameters according
to the values of N and dt.
*/

void print_graph(vector<double> v_vel, vector<double> v_cte,
                 vector<double> v_delta)
{
  plt::subplot(3, 1, 1);
  plt::title("CTE");
  plt::plot(v_cte);
  plt::subplot(3, 1, 2);
  plt::title("Delta (Radians)");
  plt::plot(v_delta);
  plt::subplot(3, 1, 3);
  plt::title("Velocity");
  plt::plot(v_vel);

  // save the figure
  plt::save(std::string("chart_") + std::to_string(PLOT_COUNTER) + std::string(".png"));
}

///////////////////////////////////////////////////////////////////////////////

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

///////////////////////////////////////////////////////////////////////////////

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s)
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos)
  {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos)
  {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

///////////////////////////////////////////////////////////////////////////////

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x)
{
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++)
  {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order)
{
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++)
  {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++)
  {
    for (int i = 0; i < order; i++)
    {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

///////////////////////////////////////////////////////////////////////////////

int main()
{
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;
  vector<double> v_vel;
  vector<double> v_cte;
  vector<double> v_delta;

  h.onMessage([&mpc, &v_vel, &v_cte, &v_delta](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                                               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2')
    {
      string s = hasData(sdata);
      if (s != "")
      {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["throttle"];

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          // Preprocessing.
          // Transforms waypoints coordinates to the cars coordinates.
          size_t n_waypoints = ptsx.size();
          auto ptsx_transformed = Eigen::VectorXd(n_waypoints);
          auto ptsy_transformed = Eigen::VectorXd(n_waypoints);
          for (unsigned int i = 0; i < n_waypoints; i++)
          {
            double dX = ptsx[i] - px;
            double dY = ptsy[i] - py;
            double minus_psi = 0.0 - psi;
            ptsx_transformed(i) = dX * cos(minus_psi) - dY * sin(minus_psi);
            ptsy_transformed(i) = dX * sin(minus_psi) + dY * cos(minus_psi);
          }

          // Fit polynomial to the points - 3rd order.
          auto coeffs = polyfit(ptsx_transformed, ptsy_transformed, 3);

          // Actuator delay in milliseconds.
          const int actuatorDelay = 100;

          // Actuator delay in seconds.
          const double delay = actuatorDelay / 1000.0;

          // Initial state.
          const double x0 = 0;
          const double y0 = 0;
          const double psi0 = 0;
          const double cte0 = coeffs[0];
          const double epsi0 = -atan(coeffs[1]);

          // State after delay.
          double x_delay = x0 + (v * cos(psi0) * delay);
          double y_delay = y0 + (v * sin(psi0) * delay);
          double psi_delay = psi0 - (v * delta * delay / mpc.Lf);
          double v_delay = v + a * delay;
          double cte_delay = cte0 + (v * sin(epsi0) * delay);
          double epsi_delay = epsi0 - (v * atan(coeffs[1]) * delay / mpc.Lf);

          // Define the state vector.
          Eigen::VectorXd state(6);
          state << x_delay, y_delay, psi_delay, v_delay, cte_delay, epsi_delay;

          // Find the MPC solution.
          auto vars = mpc.Solve(state, coeffs);

          double steer_value = vars[0] / deg2rad(25);
          double throttle_value = vars[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          // start at index 2
          for (int i = 2; i < vars.size(); i++)
          {
            // Segregate on basis of even and odd
            if (i % 2 == 0)
            {
              mpc_x_vals.push_back(vars[i]);
            }
            else
            {
              mpc_y_vals.push_back(vars[i]);
            }
          }
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          double poly_inc = 2.6;
          int num_points = 22;
          for (int i = 0; i < num_points; i++)
          {
            double x = poly_inc * i;
            next_x_vals.push_back(x);
            next_y_vals.push_back(polyeval(coeffs, x));
          }
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          // Code for plotting
          PLOT_COUNTER++;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;

          if( IS_PLOT_ENABLED )
          {
            int plot_frequency = 350;
            if (PLOT_COUNTER % plot_frequency == 0)
            {
              vector<vector<double>> tmp = mpc.GetParameters();
              cout << "KKLOG: Writing plot";
              print_graph(tmp[0], tmp[1], tmp[2]);
              cout << "KKLOG: Writing complete.";
            }
          }
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      }
      else
      {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}

///////////////////////////////////////////////////////////////////////////////