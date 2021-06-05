#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include "Eigen/Dense"

using Eigen::ArrayXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data, 
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

  vector<string> possible_labels = {"left","keep","right"};

  vector<VectorXd> gaussian_std;
  vector<VectorXd> gaussian_mean;
  vector<double> prior;
  vector<int> count;

  std::set<string> label;
  std::map<string, int> label_code;
};

#endif  // CLASSIFIER_H