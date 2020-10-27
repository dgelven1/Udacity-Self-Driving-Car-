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

using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (is_initialized) {
	  return;
  }
  else {
	num_particles = 200;  // TODO: Set the number of particles
	
	//initialize normal distribution for x, y, and theta for each component of paticles
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for(int i=0; i < num_particles; i++) {
		
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y; 
		p.theta = theta;
		p.weight = 1.0;
		
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
		
		particles.push_back(p);
        //std::cout<<"Particle = "<< p.x << std::endl;
	}
	
	is_initialized = true; 
	
  }
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
  //std::cout<<" prediction step"<<std::endl;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);   
   //determine if yaw rate is present, if not calculate simple motion model
   for(int i = 0; i < num_particles; i++) {
     double theta = particles[i].theta;
	   
	   if(fabs(yaw_rate) < 0.00001) {
		   particles[i].x += velocity * delta_t * cos(theta);
		   particles[i].y += velocity * delta_t * sin(theta);
	   }
	   else {
		   particles[i].x += velocity/yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
		   particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		   particles[i].theta += yaw_rate * delta_t;
		   
	   }
	   
	   //Add noise
	   particles[i].x += dist_x(gen);
	   particles[i].y  += dist_y(gen);
	   particles[i].theta  += dist_theta(gen);
	   
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
  //std::cout<<" data association step"<<std::endl;
   for(unsigned int i = 0; i < observations.size(); i++) {
	   LandmarkObs obs = observations[i];
	   //set min distance to very large number
	   double min_dist = numeric_limits<double>::max();
	   
	   //int mapId = -1;
       int mapId;
	   
	   for(unsigned int j = 0; j < predicted.size(); j++) {
		   LandmarkObs pred = predicted[j];
		   // get distance between obeservation and predicted
		   double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
		   
		   //determine the predicted landmark nearest to the current obeserved landmark 
		   // reset min_distance after each closer predicted landmark 
		   if(current_dist < min_dist) {
			   min_dist = current_dist;
			   mapId = pred.id;
			   
		   }
	   }
	   // update observations ID to the nearest predicted land marks id... after interating through all predicted 
	   observations[i].id = mapId;
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
   //std::cout<<" update weights step"<<std::endl;
   for( int i = 0; i < num_particles; i++) {
	   double p_x = particles[i].x;
	   double p_y = particles[i].y;
	   double p_theta = particles[i].theta;
	   
	   vector<LandmarkObs> PredictedMapLandmarks;
	   
	   for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
		   double lm_x = map_landmarks.landmark_list[j].x_f; //x-position in the map (global coordinates)
		   double lm_y = map_landmarks.landmark_list[j].y_f; //y_position
		   int lm_id = map_landmarks.landmark_list[j].id_i; // Real landmark ID
		   
		   //calc distance between particle and landmark
		   //double p_to_lm_dist = dist(p_x, p_y, lm_x, lm_y);
		   if(fabs(lm_x-p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range){
             PredictedMapLandmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y});
           }

		   /*if(p_to_lm_dist < sensor_range) {
			   // add partciles to vector
			   PredictedMapLandmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y});
             //std::cout<<"Particle in range" << std::endl;
		   }*/


	   }
	   
	   //Transform observation coordinates to map coordinates
	   vector<LandmarkObs> Transformed_obs;
	   for(unsigned int k = 0; k < observations.size(); k++) {
		   double trans_x = cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y + p_x;
		   double trans_y = sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y + p_y;
		   Transformed_obs.push_back(LandmarkObs{observations[k].id, trans_x, trans_y});

           
       }
		  
	   
        // determine dataAssociation for the predcitions and transformed observations on the current particle
        dataAssociation(PredictedMapLandmarks, Transformed_obs);
		   
        //resetting particle weight
        particles[i].weight = 1.0;
		   
        for(unsigned int j = 0; j < Transformed_obs.size(); j++) {
			//Initialize prediction vars
          	double pred_x, pred_y;
           	//Set observations equal to transformed observation coordinates
            double obs_x = Transformed_obs[j].x;
            double obs_y = Transformed_obs[j].y;
            int obs_id = Transformed_obs[j].id;
			   
            for(unsigned int k = 0; k < PredictedMapLandmarks.size(); k++) {
                if (PredictedMapLandmarks[k].id == obs_id) {
                pred_x = PredictedMapLandmarks[k].x;
                pred_y = PredictedMapLandmarks[k].y;
                
                }

            }
            
			// unpack landmark std to write calculation easier
				double std_x = std_landmark[0];
				double std_y = std_landmark[1];
				//Calculate weights for each particle
				double p_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2)))));
				/*if(p_w == 0) {
                  p_w = 0.00000001;
                }*/
				particles[i].weight *= p_w;
            //std::cout<<" particle weight = "<< p_w <<std::endl;
			}
			
	   }
  //std::cout<<" END update weights step"<<std::endl;
	   
   }
		

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   //std::cout<<" resample step"<<std::endl;
   vector<double> p_weights;
   
   //get all current weights
   //CHeck if private attribute can be accessed...if not define
   //vector<double> weights;
   
   for(int i=0; i < num_particles; i++) {
	   
	   p_weights.push_back(particles[i].weight);
   }
   //double max_w = *max_element(p_weights.begin(), p_weights.end());
  //std::cout<<"max w = "<<max_w<<std::endl;
   
   //CHECK ------------------------
  /*

   //std::uniform_real_distribution<double> distMax(0.0, max_w);
   std::discrete_distribution<int> intDist(0, num_particles-1);
   std::uniform_real_distribution<double> realDist(0.0, max_w);
   // Generating Index
   
   auto index = intDist(gen);
   //initalize beta var
   double beta = 0.0;
  
   vector<Particle> resample_p;
   // resample wheel
 
   for(int i = 0; i < num_particles; i++) {
	   beta += realDist(gen) * max_w;
	   while(beta > p_weights[index]) {
		   beta -= p_weights[index];
		   index = (index + 1) % num_particles;
	   }
	   resample_p.push_back(particles[index]);
	   
   }
   particles = resample_p;
 */
   // new resampling wheel
 // Use a discrete distribution here
  std::default_random_engine gen;
  std::discrete_distribution<> distGen(p_weights.begin(), p_weights.end());

  std::vector<Particle> resample_p;
  
  while (resample_p.size() < particles.size()) {
    int id = distGen(gen);
    resample_p.push_back(particles[id]);
  }
  
  particles = resample_p; 

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