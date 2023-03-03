#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace std;

// function to split a string by delimiter
vector < string > split(const string & s, char delimiter) {
    vector < string > tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// function to read data from a CSV file
vector < vector < double >> read_csv(string filename, char delimiter = ',') {
    ifstream file(filename);
    if (!file) {
        throw runtime_error("Unable to open file");
    }
    vector < vector < double >> data;
    string line;
    while (getline(file, line)) {
        vector < double > row;
        stringstream ss(line);
        string item;
        while (getline(ss, item, delimiter)) {
            if (!item.empty()) {
                char * endptr; // pointer to end of parsed string
                double value = strtod(item.c_str(), & endptr);
                if ( * endptr == '\0') { // check if entire string was parsed
                    row.push_back(value);
                }
            }
        }
        data.push_back(row);
    }
    return data;
}

// function to convert a vector of strings to a vector of doubles
vector < double > str2double(vector < string > v) {
    vector < double > d(v.size());
    transform(v.begin(), v.end(), d.begin(), [](const string & val) {
        return stod(val);
    });
    return d;
}

// function to calculate mean of a vector
double mean(const vector < double > & v) {
    double sum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum / v.size();
}

// function to calculate variance of a vector
double variance(const vector < double > & v, double mean) {
    double sum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        sum += pow(v[i] - mean, 2);
    }
    return sum / v.size();
}

// function to calculate standard deviation of a vector
double stdev(const vector < double > & v, double mean) {
    double
    var = variance(v, mean);
    return sqrt(var);
}

// function to calculate Gaussian probability density function
double gaussian_pdf(double x, double mean, double stdev) {
    double exponent = exp(-pow(x - mean, 2) / (2 * pow(stdev, 2)));
    double denominator = sqrt(2 * M_PI) * stdev;
    return exponent / denominator;
}

// function to train Naive Bayes model
vector < vector < double >> train_naive_bayes(const vector < vector < double >> & train_data,
    const vector < double > & train_y) {
    // separate train data by class
    vector < vector < vector < double >>> class_data(2);
    for (int i = 0; i < train_data.size(); i++) {
        int cls = (int) train_y[i];
        class_data[cls].push_back(train_data[i]);
    }
    // calculate prior probabilities
    int num_samples = train_y.size();
    double prior_0 = (double) class_data[0].size() / num_samples;
    double prior_1 = (double) class_data[1].size() / num_samples;
    // calculate means and standard deviations for each class and predictor
    int num_predictors = train_data[0].size();
    vector < vector < double >> means(2, vector < double > (num_predictors));
    vector < vector < double >> stdevs(2, vector < double > (num_predictors));
    for (int cls = 0; cls < 2; cls++) {
        for (int j = 0; j < num_predictors; j++) {
            vector < double > values;
            for (int i = 0; i < class_data[cls].size(); i++) {
                values.push_back(class_data[cls][i][j]);
            }
            double mean_value = mean(values);
            double stdev_value = stdev(values, mean_value);
            means[cls][j] = mean_value;
            stdevs[cls][j] = stdev_value;
        }
    }
    // return model parameters
    vector < vector < double >> model_params = {
        means[0],
        stdevs[0],
        means[1],
        stdevs[1],
        {
            prior_0,
            prior_1
        }
    };
    return model_params;
}

// function to predict class using Naive Bayes model
double predict_naive_bayes(const vector < vector < double >> & model_params,
    const vector < double > & x) {
    // unpack model parameters
    vector < double > mean_0 = model_params[0];
    vector < double > stdev_0 = model_params[1];
    vector < double > mean_1 = model_params[2];
    vector < double > stdev_1 = model_params[3];
    double prior_0 = model_params[4][0];
    double prior_1 = model_params[4][1];
    // calculate likelihoods for each predictor
    double likelihood_0 = 1.0;
    double likelihood_1 = 1.0;
    for (int j = 0; j < x.size(); j++) {
        double pdf_0 = gaussian_pdf(x[j], mean_0[j], stdev_0[j]);
        double pdf_1 = gaussian_pdf(x[j], mean_1[j], stdev_1[j]);
        likelihood_0 *= pdf_0;
        likelihood_1 *= pdf_1;
    }
    // calculate posterior probabilities
    double posterior_0 = likelihood_0 * prior_0;
    double posterior_1 = likelihood_1 * prior_1;
    // return predicted class
    return posterior_1 > posterior_0 ? 1.0 : 0.0;
}

// function to calculate accuracy, sensitivity, and specificity of predictions
vector < double > evaluate_predictions(const vector < vector < double >> & test_data,
    const vector < double > & test_y,
        const vector < vector < double >> & model_params) {
    int num_correct = 0;
    int num_true_positives = 0;
    int num_false_positives = 0;
    int num_true_negatives = 0;
    int num_false_negatives = 0;
    for (int i = 0; i < test_data.size(); i++) {
        double y_true = test_y[i];
        double y_pred = predict_naive_bayes(model_params, test_data[i]);
        if (y_true == y_pred) {
            num_correct++;
            if (y_true == 1.0) {
                num_true_positives++;
            } else {
                num_true_negatives++;
            }
        } else {
            if (y_true == 1.0) {
                num_false_negatives++;
            } else {
                num_false_positives++;
            }
        }
    }
    double accuracy = (double) num_correct / test_data.size();
    double sensitivity = (double) num_true_positives / (num_true_positives + num_false_negatives);
    double specificity = (double) num_true_negatives / (num_true_negatives + num_false_positives);
    vector < double > metrics = {
        accuracy,
        sensitivity,
        specificity
    };
    return metrics;
}

int main() {
    // read data from CSV file
    vector < vector < double >> data = read_csv("titanic_project.csv");
    // convert data to double and separate predictors and response
    vector < vector < double >> predictors(data.size() - 1);
    vector < double > response(data.size() - 1);
    for (int i = 1; i < data.size(); i++) {
        vector < double > row(data[i].size());
        for (int j = 0; j < data[i].size(); j++) {
            row[j] = stod(to_string(data[i][j]));
        }
        predictors[i - 1] = {
            row[2],
            row[3],
            row[4]
        };
        response[i - 1] = row[1];
    }
    // split data into train and test sets
    int num_train = 800;
    vector < vector < double >> train_data(predictors.begin(), predictors.begin() + num_train);
    vector < double > train_y(response.begin(), response.begin() + num_train);
    vector < vector < double >> test_data(predictors.begin() + num_train, predictors.end());
    vector < double > test_y(response.begin() + num_train, response.end());
    // fit Naive Bayes model on train data
    vector < vector < double >> model_params = train_naive_bayes(train_data, train_y);
    // evaluate model on test data
    vector < double > metrics = evaluate_predictions(test_data, test_y, model_params);
    // output metrics
    cout << "Accuracy: " << metrics[0] << endl;
    cout << "Sensitivity: " << metrics[1] << endl;
    cout << "Specificity: " << metrics[2] << endl;
    return 0;
}