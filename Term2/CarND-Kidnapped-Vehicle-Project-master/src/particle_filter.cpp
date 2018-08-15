/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

///////////////////////////////////////////////////////////////////////////////

#define SMALL_VAL 0.00001

///////////////////////////////////////////////////////////////////////////////

using namespace std;

///////////////////////////////////////////////////////////////////////////////

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Check if already initialized
  if (initialized())
  {
    return;
  }

  // Initializing the number of particles
  //  @KK play with different values to see the variations in the final value
  num_particles = 100;

  // Extracting standard deviations
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Creating normal distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Generate particles with normal distribution with mean on GPS values.
  for (int i = 0; i < num_particles; i++)
  {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  // The filter is now initialized.
  is_initialized = true;
}

///////////////////////////////////////////////////////////////////////////////

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Pick the standard deviations values
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Normal distributions from the above values
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  // Calculate for the new state.
  for (int i = 0; i < num_particles; i++)
  {
    double theta = particles[i].theta;

    if (fabs(yaw_rate) < SMALL_VAL)
    { // When yaw is not changing.
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
      // yaw continue to be the same.
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add noise as suggested in the quiz solution
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

///////////////////////////////////////////////////////////////////////////////

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (auto it_observation = std::begin(observations);
       it_observation != std::end(observations); ++it_observation)
  {
    double minDistance = numeric_limits<double>::max();

    // Init id for invalid or not found entries
    int mapId = -1;

    for (auto it_prediction = std::begin(predicted);
         it_prediction != std::end(predicted); ++it_prediction)
    {
      // Compute distance
      double xDistance = it_observation->x - it_prediction->x;
      double yDistance = it_observation->y - it_prediction->y;
      double distance = xDistance * xDistance + yDistance * yDistance;

      // If distance is less the min threshold, fetch mapId map(measurement->landmark)
      // and update the min distance
      if (distance < minDistance)
      {
        minDistance = distance;
        mapId = it_prediction->id;
      }
    }

    // Update the observation identifier.
    it_observation->id = mapId;
  }
}

///////////////////////////////////////////////////////////////////////////////

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  double std_landmark_range = std_landmark[0];
  double std_landmark_bearing = std_landmark[1];

  // Using the code snippets and logic from the lecture quizes
  // Particle weight calculation section

  // iterate until the number of particles
  for (int i = 0; i < num_particles; i++)
  {
    // Fetch the attributes of the particle
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    // Search for the landmarks within the particle's range.
    double sensor_range_2 = sensor_range * sensor_range;
    vector<LandmarkObs> in_range_landmarks;

    for (auto it_map_landmarks = std::begin(map_landmarks.landmark_list);
         it_map_landmarks != std::end(map_landmarks.landmark_list); ++it_map_landmarks)
    {
      // Fetch landmark attributes
      float landmarkX = it_map_landmarks->x_f;
      float landmarkY = it_map_landmarks->y_f;
      int id = it_map_landmarks->id_i;
      double dX = x - landmarkX;
      double dY = y - landmarkY;

      if (dX * dX + dY * dY <= sensor_range_2)
      {
        // Fill the vector containing the landmarks within range
        in_range_landmarks.push_back(LandmarkObs{id, landmarkX, landmarkY});
      }
    }

    // Transform observation coordinates.
    vector<LandmarkObs> mapped_observations;
    for (auto it_observation = std::begin(observations);
         it_observation != std::end(observations); ++it_observation)
    {
      double xx = cos(theta) * it_observation->x - sin(theta) * it_observation->y + x;
      double yy = sin(theta) * it_observation->x + cos(theta) * it_observation->y + y;
      mapped_observations.push_back(LandmarkObs{it_observation->id, xx, yy});
    }

    // Observation association to landmark
    dataAssociation(in_range_landmarks, mapped_observations);

    // Reset the weights
    particles[i].weight = 1.0;

    // Calculate weights.
    for (auto it_mapped_observation = std::begin(mapped_observations);
         it_mapped_observation != std::end(mapped_observations); ++it_mapped_observation)
    {
      double observationX = it_mapped_observation->x;
      double observationY = it_mapped_observation->y;

      int landmarkId = it_mapped_observation->id;

      double landmarkX, landmarkY;
      unsigned int k = 0;
      unsigned int nLandmarks = in_range_landmarks.size();
      bool found = false;
      while (!found && k < nLandmarks)
      {
        if (in_range_landmarks[k].id == landmarkId)
        {
          found = true;
          landmarkX = in_range_landmarks[k].x;
          landmarkY = in_range_landmarks[k].y;
        }
        k++;
      }

      // Calculating weight.
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;

      double weight = (1 / (2 * M_PI * std_landmark_range * std_landmark_bearing)) * exp(-(dX * dX / (2 * std_landmark_range * std_landmark_range) * +(dY * dY / (2 * std_landmark_bearing * std_landmark_bearing))));
      if (weight == 0)
      {
        // If 0, assing the small value
        particles[i].weight *= SMALL_VAL;
      }
      else
      {
        particles[i].weight *= weight;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Use the weights and max weight.
  vector<double> weights;

  // Using the code snippets from the lecture quizzes
  // Get the min value
  double max_weight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight)
    {
      max_weight = particles[i].weight;
    }
  }

  // Create the distributions using the above values
  uniform_real_distribution<double> distDouble(0.0, max_weight);
  uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generate index using the random generator.
  int index = distInt(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; i++)
  {
    beta += distDouble(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

///////////////////////////////////////////////////////////////////////////////

Particle ParticleFilter::SetAssociations(Particle &particle,
                                         const std::vector<int> &associations,
                                         const std::vector<double> &sense_x,
                                         const std::vector<double> &sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous entries
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

///////////////////////////////////////////////////////////////////////////////

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

///////////////////////////////////////////////////////////////////////////////