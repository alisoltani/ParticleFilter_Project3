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

	num_particles = 20;

    default_random_engine gen;

	// Generate and add noise to particles
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
    }

    is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // First precalculate some terms to save time
	double v_dt = velocity * delta_t;
	double yaw_dt = yaw_rate * delta_t;
	double v_yaw = velocity/yaw_rate;

	default_random_engine gen;

    for (Particle& particle: particles) {

    	double pred_x;
        double pred_y;
        double pred_theta;

		if (fabs(yaw_rate) < EPS) { // different expressions depending on zero yaw_rate
			pred_theta = particle.theta;
			pred_x = particle.x + v_dt * cos(particle.theta);
			pred_y = particle.y + v_dt * sin(particle.theta);
		}
		else
		{
			pred_theta = particle.theta + yaw_dt;
			pred_x = particle.x + v_yaw * (sin(pred_theta) - sin(particle.theta));
			pred_y = particle.y + v_yaw * (cos(particle.theta) - cos(pred_theta));
		}

		// Generate and add noise to particles
		double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
		std_x = std_pos[0];
		std_y = std_pos[1];
		std_theta = std_pos[2];

		// generate the normal distributions with zero mean
		normal_distribution<double> dist_x(pred_x, std_x);
		normal_distribution<double> dist_y(pred_y, std_y);
		normal_distribution<double> dist_theta(pred_theta, std_theta);
		// add noise to particles
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}

}

// See comment in ParticleFilter::closestLandmarkLocation() implementation below
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {


    for (Particle& particle: particles) {
        particle.weight = 1;
    	double exp_term = 0.0;

        vector<int> associations;
        vector<double> senseX;
        vector<double> senseY;

        for (LandmarkObs& observation: observations) { // for all observations

            // Transform into map coordinates for each particle
            LandmarkObs transformed_observation;

            transformed_observation.x = particle.x + (observation.x*cos(particle.theta) - observation.y*sin(particle.theta));
            transformed_observation.y = particle.y + (observation.x*sin(particle.theta) + observation.y*cos(particle.theta));

            // Find the nearest landmark to observations that is in sensor range
            double nearest_landmark_distance = sensor_range;
            double minx = 0;
            double miny = 0;
            int nearest_landmark_id = 0;


            for (auto& landmark: map_landmarks.landmark_list) {
                double distance = dist(landmark.x_f, landmark.y_f, transformed_observation.x, transformed_observation.y);
                if (distance < nearest_landmark_distance) {
                    nearest_landmark_distance = distance;
                    minx = landmark.x_f;
                    miny = landmark.y_f;
                    nearest_landmark_id = landmark.id_i;
                }
            }

            if (nearest_landmark_distance!=sensor_range) {
            	// only calculate the exp power term to save on computation
            	exp_term += pow(transformed_observation.x - minx, 2) / pow(std_landmark[0], 2) + pow(transformed_observation.y - miny, 2) / pow(std_landmark[1], 2);

                associations.push_back(nearest_landmark_id);
                senseX.push_back(transformed_observation.x);
                senseY.push_back(transformed_observation.y);
            }
        }

        particle.weight *= exp(-exp_term) / (2.0 * M_PI * std_landmark[0] * std_landmark[0]); // take exp here instead to speed up things.

        particle = SetAssociations(particle, associations, senseX, senseY);

    }
}

void ParticleFilter::resample() {
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;

    std::vector<double> weights;
    for (Particle& p: particles) {
        weights.push_back(p.weight);
    }

    discrete_distribution<int> distribution(weights.begin(), weights.end());

    vector<Particle> resample_particles;
    // simple resampling
    // todo: improve the resampling to better reflect probabilities
    for (int i = 0; i < num_particles; ++i) {
        resample_particles.push_back(particles[distribution(gen)]);
    }

    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    /// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    /// associations: The landmark id that goes along with each listed association
    /// sense_x: the associations x mapping already converted to world coordinates
    /// sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
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
