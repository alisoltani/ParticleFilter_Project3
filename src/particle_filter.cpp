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

#define EPS 0.001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;
	double init_weight_ = 1.0;

	particles.resize(num_particles);
	weights.resize(num_particles);

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Adding noise to each particle and setting all weights to 1
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = init_weight_;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// First precalculate some terms to save time
	double v_dt = velocity * delta_t;
	double yaw_dt = yaw_rate * delta_t;
	double v_yaw = velocity/yaw_rate;


	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < EPS) {
			particles[i].x += v_dt * cos(particles[i].theta);
			particles[i].y += v_dt * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += v_yaw * (sin(particles[i].theta + yaw_dt) - sin(particles[i].theta) );
			particles[i].y += v_yaw * (cos(particles[i].theta + yaw_dt) - cos(particles[i].theta) );
			particles[i].theta += particles[i].theta + yaw_dt;
		}

		// Generate and add noise to particles
		default_random_engine gen;

		double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
		std_x = std_pos[0];
		std_y = std_pos[1];
		std_theta = std_pos[2];

		// generate the normal distributions with zero mean
		normal_distribution<double> dist_x(0.0, std_x);
		normal_distribution<double> dist_y(0.0, std_y);
		normal_distribution<double> dist_theta(0.0, std_theta);
		// add noise to particles
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// This implements the nearest neighbor method by going through all combinations to find
	// the minimum distance

	for (LandmarkObs& observation: observations) {
		int nearest_landmark = 0;
		double nearest_landmark_distance = INFINITY;

		for (LandmarkObs& predict: predicted) {
			double diff = dist(predict.x, predict.y, observation.x, observation.y);

			if (diff < nearest_landmark_distance){
				nearest_landmark_distance = diff;
				nearest_landmark = predict.id;
			}
		}

		observation.id = nearest_landmark;

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	for (int i=0; i < num_particles; i++) {

		// TODO: Make into separate method
		///////////////////////////////////
		// Transform coordinates
		vector<LandmarkObs> transformed_observations;
		for (LandmarkObs& observation: observations) {
			LandmarkObs transformed_observation;
			transformed_observation.id = observation.id;
			transformed_observation.x = observation.x * cos(particles[i].theta) - observation.y * sin(particles[i].theta) + particles[i].x;
			transformed_observation.y = observation.x * sin(particles[i].theta) + observation.y * cos(particles[i].theta) + particles[i].y;
			transformed_observations[i] = transformed_observation;
		}

        // Obtain predicted landmark list
        vector<LandmarkObs> predict;
        for (Map::single_landmark_s& landmark : map_landmarks.landmark_list) {

          if (dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y) <= sensor_range) {
            LandmarkObs prediction {landmark.id_i, landmark.x_f, landmark.y_f};
            predict.push_back(prediction);
          }
        }

        // associate observations with predictions
        dataAssociation(predict, transformed_observations);
		///////////////////////////////////

        // Update the weights for each particle
        particles[i].weight = 1;

        vector<int> associations;
        vector<double> senseX;
        vector<double> senseY;

        for (LandmarkObs& obs: transformed_observations) {
        	// Get current prediction
        	Map::single_landmark_s prediction = map_landmarks.landmark_list[obs.id - 1];

            // Diff between measurement and prediction
            double dx = obs.x - prediction.x_f;
            double dy = obs.y - prediction.y_f;

            // Calculate the new weight
            double new_weight = 1 / (M_PI * 2 * std_landmark[0] * std_landmark[1]) *
                  exp(-1 * (pow(dx, 2) / pow(std_landmark[0], 2) + pow(dy, 2) / pow(std_landmark[1], 2)));

            // Multiply running product of weights by the new weight
            particles[i].weight *= new_weight;
          }
          weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> distribution {weights.begin(), weights.end()};

    vector<Particle> new_particles;
    for (int i=0; i < num_particles; i++){
        int new_particle_index = distribution(gen);
        Particle new_particle = particles[new_particle_index];
        new_particles.push_back(new_particle);
    }

    particles = new_particles;

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
