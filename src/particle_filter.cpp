/*
 * particle_filter.cpp
 *
 *  Created on: October 27, 2017
 *      Author: Olga Oleksyuk https://github.com/ooleksyuk
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

using namespace std;

const int NUMBER_OF_PARTICLES = 100; //50; //300;
const double INITIAL_WEIGHT = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = NUMBER_OF_PARTICLES;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = INITIAL_WEIGHT;

    weights.push_back(INITIAL_WEIGHT);
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  const double THRESHOLD = 0.001;
  const double INITIAL_VAL = 0.0;
  const double MOVING_STRAIGHT = abs(yaw_rate) < THRESHOLD;

  normal_distribution<double> dist_x(INITIAL_VAL, std_pos[0]);
  normal_distribution<double> dist_y(INITIAL_VAL, std_pos[1]);
  normal_distribution<double> dist_theta(INITIAL_VAL, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    const double theta = particles[i].theta;
    const double noise_x = dist_x(gen);
    const double noise_y = dist_y(gen);
    const double noise_theta = dist_theta(gen);

    if (MOVING_STRAIGHT) {
      particles[i].x += velocity * delta_t * cos(theta) + noise_x;
      particles[i].y += velocity * delta_t * sin(theta) + noise_y;
      particles[i].theta += noise_theta;
    } else {
      const double phi_theta = theta + delta_t * yaw_rate;
      particles[i].x += velocity / yaw_rate * (sin(phi_theta) - sin(theta)) + noise_x;
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(phi_theta)) + noise_y;
      particles[i].theta = phi_theta + noise_theta;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the
  // observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  const double ERROR = 1.0e99;
  for (int i = 0; i < observations.size(); i++) {
    int current_observation;
    double current_error = ERROR;

    for (int j = 0; j < predicted.size(); j++) {
      const double delta_x = predicted[j].x - observations[j].x;
      const double delta_y = predicted[j].y - observations[j].y;
      const double error = delta_x * delta_x + delta_y + delta_y;

      if (error < current_error) {
        current_observation = j;
        current_error = error;
      }
    }
    observations[i].id = current_observation;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
    const double p_x = particles[i].x;
    const double p_y = particles[i].y;
    const double p_theta = particles[i].theta;

    vector<LandmarkObs> landmarks_in_range;
    vector<LandmarkObs> transformed_landmarks;

    /*******************************************************************
     * STEP 1: Transform coordinates of landmarks from vehicle perspective to
     * relevant to the position of the particle, pretend that the vehicle is
     * exactly where the particle is and is oriented the same way as particle
     */
    for (int j = 0; j < observations.size(); j++) {
      double obs_id = observations[j].id;
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

      double transformed_x = p_x + obs_x * cos(p_theta) - obs_y * sin(p_theta);
      double transformed_y = p_y + obs_y * cos(p_theta) + obs_x * sin(p_theta);

      LandmarkObs transformed_landmark;
      transformed_landmark.id = obs_id;
      transformed_landmark.x = transformed_x;
      transformed_landmark.y = transformed_y;

      transformed_landmarks.push_back(transformed_landmark);
    }
    /**
     * STEP 2: Find landmarks within sensor range
     */
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;

      double distance_x = landmark_x - p_x;
      double distance_y = landmark_y - p_y;
      double distance_error = sqrt(distance_x * distance_x + distance_y * distance_y);

      if (distance_error < sensor_range) {
        LandmarkObs landmark_in_range;
        landmark_in_range.id = landmark_id;
        landmark_in_range.x = landmark_x;
        landmark_in_range.y = landmark_y;

        landmarks_in_range.push_back(landmark_in_range);
      }
    }

    /**
     * STEP 3: Associate landmark in range by `id` with observation landmark
     * this function modifies `std::vector<LandmarkObs>` observations
     * Note: All landmarks are in map coordinates
     *       All observations are in map coordinates
     */
      dataAssociation(landmarks_in_range, transformed_landmarks);

    /**
     * STEP 4: Compare each observation by vehicle `map_observations`
     * to observation by particle landmarks_in_range
     * Update particle weight based on result of the comprehension
     */
    double weight = INITIAL_WEIGHT;
    const double num_x = 1 / (2 * std_x * std_x);
    const double num_y = 1 / (2 * std_y * std_y);
    const double denominator = sqrt(2.0 * M_PI * std_x * std_y);

    for (int j = 0; j < transformed_landmarks.size(); j++) {
      int obs_id = transformed_landmarks[j].id;
      double obs_x = transformed_landmarks[j].x;
      double obs_y = transformed_landmarks[j].y;

      double predicted_x = landmarks_in_range[obs_id].x;
      double predicted_y = landmarks_in_range[obs_id].y;

      double d_x = obs_x - predicted_x;
      double d_y = obs_y - predicted_y;

      double a_x = num_x * d_x * d_x;
      double a_y = num_y * d_y * d_y;

      double exponential = exp(-(a_x + a_y)) / denominator;
      weight *= exponential;
    }

    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> sampled_particles;
  default_random_engine gen;
  discrete_distribution<int> index(weights.begin(), weights.end());

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
    const int idx = index(gen);
    Particle particle;
    particle.id = idx;
    particle.x = particles[idx].x;
    particle.y = particles[idx].y;
    particle.theta = particles[idx].theta;
    particle.weight = INITIAL_WEIGHT;

    sampled_particles.push_back(particle);
  }
  particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
   particle.sense_x = sense_x;
   particle.sense_y = sense_y;

   return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
