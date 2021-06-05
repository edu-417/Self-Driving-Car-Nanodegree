#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using std::string;
using std::vector;

// Initializes GNB
GNB::GNB() {
  /**
   * TODO: Initialize GNB, if necessary. May depend on your implementation.
   */
  
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * TODO: Implement the training function for your classifier.
   */
    label = std::set<string>(labels.begin(), labels.end());

    int id = 0;
    for (std::set<string>::iterator it=label.begin(); it != label.end(); ++it) {
        label_code[*it] = id++;
    }

    int label_size = label.size();
    int feature_size = data[0].size();

    gaussian_std = vector<VectorXd>(label_size, VectorXd::Zero(feature_size));
    gaussian_mean = vector<VectorXd>(label_size, VectorXd::Zero(feature_size));
    prior = vector<double> (label_size);
    count = vector<int> (label_size, 0); //can be used for online training

    for (unsigned int i = 0; i < data.size(); ++i) {
        int code = label_code[labels[i]];
        gaussian_mean[code] += VectorXd::Map(data[i].data(), data[i].size());
        count[code]++;
    }

    for (int i = 0; i < label_size; ++i) {
        gaussian_mean[i] /= count[i];
        prior[i] = count[i] * 1.0 / data.size();
    }

    for (unsigned int i = 0; i < data.size(); ++i) {
        int code = label_code[labels[i]];
        VectorXd data_vector = VectorXd::Map(data[i].data(), data[i].size());
        VectorXd residual = (data_vector - gaussian_mean[code]);
        residual = (residual.array() * residual.array());
        gaussian_std[code] += residual;
    }

    for (int i =0; i < label_size; ++i) {
        gaussian_std[i] /= count[i];
        gaussian_std[i] = gaussian_std[i].array().sqrt();
    }
}

inline double gaussian(double x, double ux, double sigmaX) {
    double gaussNorm = 1/ (sqrt( 2 * M_PI) * sigmaX);
    double exponent = (x - ux) * (x - ux) / ( 2 * sigmaX * sigmaX );
    double prob =  gaussNorm * exp(-0.5 * exponent) ;

    return prob;
}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * TODO: Complete this function to return your classifier's prediction
   */

    int label_size = label.size();
    vector<double> predictions(label_size);

    for (int i = 0; i < label_size; ++i) {
        predictions[i] = prior[i];
        for (int j = 0; j < gaussian_mean[i].size(); ++j){
            double prob = gaussian(sample[j], gaussian_mean[i][j], gaussian_std[i][j]);
            predictions[i] *= prob;
        }
    }

    int max_id = -1;
    double max_prob = 0;
    for (int i = 0; i < label_size; ++i){
        if(predictions[i] > max_prob){
            max_prob = predictions[i];
            max_id = i;
        }
    }

    std::set<string>::iterator it = label.begin();
    for(int i=0; i < max_id; ++i) it++;

    return (*it);
}