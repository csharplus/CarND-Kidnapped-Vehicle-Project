/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles
  
  // Create random Gaussian noises for x, y and theta
  normal_distribution<double> nd_x(0, std[0]);
  normal_distribution<double> nd_y(0, std[1]);
  normal_distribution<double> nd_theta(0, std[2]);

  // Initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x + nd_x(gen);
    p.y = y + nd_y(gen);
    p.theta = theta + nd_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Random Gaussian noises
  normal_distribution<double> nd_x(0, std_pos[0]);
  normal_distribution<double> nd_y(0, std_pos[1]);
  normal_distribution<double> nd_theta(0, std_pos[2]);
  
  // Update predicted values for all particles
  for (int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;
    if (fabs(yaw_rate) < 0.0001) {
      double position_change = velocity * delta_t;
      particles[i].x += position_change * cos(theta);
      particles[i].y += position_change * sin(theta);
    } else {
      double yaw_change = yaw_rate * delta_t;
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_change) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_change));
      particles[i].theta += yaw_change;
    }

    particles[i].x += nd_x(gen);
    particles[i].y += nd_y(gen);
    particles[i].theta += nd_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {
    double min_dist = numeric_limits<double>::max();
    int pred_id = 0;
    
    // Iterate through all the predictions to find out the closest to the current observation
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // Adjust predicted landmark based on distance
      if (current_dist < min_dist) {
        min_dist = current_dist;
        pred_id = predicted[j].id;
      }
    }

    // Link observation to the closest prediction
    observations[i].id = pred_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // Find the map landmark locations predicted within sensor range of the particle
    vector<LandmarkObs> pred_in_range;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float l_x = map_landmarks.landmark_list[j].x_f;
      float l_y = map_landmarks.landmark_list[j].y_f;
      int l_id = map_landmarks.landmark_list[j].id_i;
      
      // Keep only landmarks within sensor range
      if (dist(p_x, p_y, l_x, l_y) <= sensor_range) {
        pred_in_range.push_back(LandmarkObs{ l_id, l_x, l_y });
      }
    }

    // Get observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      transformed_obs.push_back(
        LandmarkObs{ observations[j].id, 
                    cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x,
                    sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y });
    }

    dataAssociation(pred_in_range, transformed_obs);

    // Reset weight
    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      double o_x, o_y, pr_x, pr_y;
      o_x = transformed_obs[j].x;
      o_y = transformed_obs[j].y;

      int associated_pred_id = transformed_obs[j].id;

      // Obtain coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < pred_in_range.size(); k++) {
        if (pred_in_range[k].id == associated_pred_id) {
          pr_x = pred_in_range[k].x;
          pr_y = pred_in_range[k].y;
        }
      }

      // Calculate weight with multi-variate Gaussian
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*std_x*std_y) ) * 
        exp( - (((pr_x-o_x)*(pr_x-o_x)/(2*std_x*std_x) + ((pr_y-o_y)*(pr_y-o_y)/(2*std_y*std_y))) ));

      // Update weight
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Get all the particle weights
  weights.clear();
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // Declare resampled particles
  vector<Particle> resampled_particles(num_particles);

  // Generate resampled particles with random distribution based on weights
  random_device rd;
  default_random_engine rand_gen(rd());
  discrete_distribution<> disc_dist(weights.begin(), weights.end());
  for(int i=0; i<num_particles; i++){
    resampled_particles[i] = particles[disc_dist(rand_gen)];
  }

  // Set particles to be resampled particles 
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
