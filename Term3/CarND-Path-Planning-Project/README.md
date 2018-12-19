# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

# Implementation

**Result**

* The car was able to complete more than 3 laps of the track without any collison. It never left the lane other than doing lane change.
* There was no exceeding jerk or acceleration message.
* The car was driving with 49.5 KPH of maximum speed, only slowing down in traffic and doing lane change if and when possible.

![Video](images/video.gif)

Full video can be found at [Video](./video/video.mp4)

**Code Structure**

The code structure is as below:
1. main.cpp : Contains the main behaviour and changes related to the project.
2. constants.h : Is a header file having constants declared within.
3. Spline.h : Library used for generating spines using the provided anchor points

**Detailed walkthrough of the changes within main.cpp**

1. The default lane for the ego car ( our car ) is set as middle lane, i.e. 1.
2. We receive list of target cars in our side of the street using the provided sensor fusion data. This information is used to check on which sides of the ego car we have some target objects. [line 280](./src/main.cpp#L280)
3. We also use sensor fusion information to know the lanes of the target cars. [line 289](./src/main.cpp#L289)
4. We then calculate the speed and estimate the position of car using previous trajectory. [line 314](./src/main.cpp#L314)
5. We estimate if the target car is on left, right or same lane as ego car. The threshold distance to look is set to 30 meters. [line 316](./src/main.cpp#L316)
6. Once we have the information on the target vehicles, we try to switch lanes. This is done using the simple fsm described in the lecture and workshop.

**Finite State Machine**

![FSM](images/fsm.png)

* As seen in the picture above. The FSM can suggest either Lane Keep state or prepare to do lane change, either to left or right ( depends on the information received from the sensor fusion ).

* Each state check itself for all possible state transitions.
* At the Lane Keep state, the ego car will move on its lane as long as there is no other vehicle in front. Apart from this, the state will ensure the ego car does not exceed the allowed speed limit.
* The Prepare Lane Change Left/Right comes into play if there is no other car in left i.e. 0 or right i.e. 2 lane.
* Once Prepare Lane Change is set, the state transits to Lane Change. The ego car will do the lane change here.
* After lane change, the ego car will return to keep lane state.

**Trajectory Generation**

* The path generation is done by generating splines using the spline.h header.
* Splines are piecewise polymetric curve. The path planner used here is a spline mathematical function used for curve fitting the map coordinates.
* The splines are useful as they generate quite smooth curves which create nice smoothened acceleration and jerk profiles. This is done at [line 432](./src/main.cpp#L432)
* After the spline are created, we have to recompute the current map points using the generated curve.
* To recompute we use equidistant points that will keep our desired speed. It should look like below:
  
  ![equisitant](images/equidistant.PNG)
* Now we compute the coordinates using the spline
* After we have the coordinates, we shift the orientation back.

   
### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

### Goals
In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

**Successful cmake and make**

![compile](images/compile.PNG)

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

