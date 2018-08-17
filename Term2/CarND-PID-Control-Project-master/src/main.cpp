#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"

// for convenience
using json = nlohmann::json;

///////////////////////////////////////////////////////////////////////////////

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

///////////////////////////////////////////////////////////////////////////////

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s)
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos)
  {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos)
  {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  uWS::Hub h;

  PID steering_pid;
  PID vel_pid;
  // TODO: Initialize the pid variable.
  double steering_gain = 2.9;

  // Initialize the PID controller for steering angle values
  double steering_p = 0.09671100;
  double steering_i = 0.0000171;
  double steering_d = 1.8992100;

  steering_pid.Init(steering_p, steering_i, steering_d);
  //steering_pid.Init(0, 0, 1.9);

  // Initialize the PID Controller for speed values
  double speed_p = 0.1092310;
  double speed_i = 0.0007210;
  double speed_d = 0.8510000;

  vel_pid.Init(speed_p, speed_i, speed_d);

  h.onMessage([&steering_pid, &vel_pid, &steering_gain, &argv, &argc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "")
      {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());

          // Update error for derivative calculation
          steering_pid.UpdateError(cte);
          
          // Calculate PID value for the steering angle
          double steer_angle = steering_pid.ControlSignal(cte);

          // Calculate PD value for throttle
          // Set target speed
          
          // PID for the speed is invoked here
          float speed_lower_threshold = 30.0;
          float speed_higher_threshold = 55.0;
          double target_speed = speed_higher_threshold * (1.0 - steering_gain * fabs(steer_angle)) + speed_lower_threshold;
          std::cout << "KKLOG: Target speed is => " << target_speed << " kph" << std::endl;

          // speed error => cte
          double speed_error = speed - target_speed;

          // Calculate speed value
          vel_pid.UpdateError(speed_error);
          double speed_value = vel_pid.ControlSignal(speed_error);
          // Push calculated values to the simulator
          json msgJson;
          msgJson["steering_angle"] = steer_angle;
          msgJson["throttle"] = speed_value;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

          if (argc > 1 && strcmp(argv[1], "twiddle") == 0)
          {
            std::cout << "KKLOG: Twiddle activated." << std::endl;
            steering_pid.twiddle();
            vel_pid.twiddle();
          }
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

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
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

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
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
