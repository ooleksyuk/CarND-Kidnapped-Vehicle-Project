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

using namespace std;
const int NUMBER_OF_PARTICLES = 100; //50; //300;
const double INITIAL_WEIGHT = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  this->num_particles = NUMBER_OF_PARTICLES;
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

    this->weights.push_back(INITIAL_WEIGHT);
    this->particles.push_back(INITIAL_WEIGHT);
  }
  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  const double THRESHOLD = 0.001;
  const double INITIAL_COORDINATE = 0.0;
  const double MOVING_STRAIGHT = fabs(yaw_rate) < THRESHOLD;
  const double v_theta = MOVING_STRAIGHT ? velocity * delta_t : velocity / yaw_rate;
  const double delta_theta = delta_t * yaw_rate;

  normal_distribution dist_x(INITIAL_COORDINATE, std_pos[0]);
  normal_distribution dist_y(INITIAL_COORDINATE, std_pos[1]);
  normal_distribution dist_theta(INITIAL_COORDINATE, std_pos[2]);

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
    const double theta = this->particles[i].theta;
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double noise_x = dist_x(gen);
    const double noise_y = dist_y(gen);
    const double noise_theta = dist_theta(gen);

    if (MOVING_STRAIGHT) {
      this->particles[i].x = v_theta * cos_theta + noise_x;
      this->particles[i].y = v_theta * sin_theta + noise_y;
      this->particles[i].theta += noise_theta;
    } else {
      const double phi_theta = theta + delta_theta;
      this->particles[i].x = v_theta * (sin(phi_theta) - sin_theta) + noise_x;
      this->particles[i].y = v_theta * (cos_theta - cos(phi_theta)) + noise_y;
      this->particles[i].theta = phi_theta + noise_theta;
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
      const error = delta_x * delta_x + delta_y + delta_y;

      if (current_error < error) {
        current_observation = j;
        current_error = error;
      }
    }
    observations[i].id = current_observation;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> sampled_particles;
  default_random_engine gen;
  discrete_distribution<int> index(this->weights.begin(), this->weights.end());

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
    const int idx = index(gen);
    Particle particle;
    particle.id = idx;
    particle.x = this->particles[idx].x;
    particle.x = this->particles[idx].y;
    particle.theta = this->particles[idx].theta;
    particle.weight = INITIAL_WEIGHT;

    sampled_particles.push_back(particle);
  }

  this->particles.push_back(sampled_particles);
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
