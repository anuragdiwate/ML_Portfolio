#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

vector < vector < double >> read_csv(string filename) {
    ifstream infile(filename.c_str());
    if (!infile.good()) {
        cerr << "Error: could not open file " << filename << endl;
        exit(1);
    }

    vector < vector < double >> data;
    string line;
    while (getline(infile, line)) {
        vector < double > row;
        istringstream iss(line);
        string token;
        while (getline(iss, token, ',')) {
            try {
                row.push_back(stod(token));
            } catch (const std::invalid_argument & e) {
                row.push_back(0.0);
            }
        }
        data.push_back(row);
    }

    return data;
}
// function to split a string by a delimiter
vector < string > split(string str, char delimiter) {
    vector < string > internal;
    stringstream ss(str);
    string tok;
    while (getline(ss, tok, delimiter)) {
        internal.push_back(tok);
    }
    return internal;
}

// function to perform logistic regression
vector < double > logistic_regression(vector < vector < double >> data, int num_iterations, double learning_rate) {
    int num_features = data[0].size() - 1; // number of predictors (excluding the target variable)
    vector < double > coefficients(num_features, 0.0); // initialize coefficients to 0
    int num_samples = data.size(); // number of samples
    for (int i = 0; i < num_iterations; i++) {
        double cost = 0.0;
        vector < double > gradient(num_features, 0.0); // initialize gradient to 0
        for (int j = 0; j < num_samples; j++) {
            double y = data[j][0]; // target variable
            vector < double > x(num_features, 0.0); // predictors
            for (int k = 1; k <= num_features; k++) {
                x[k - 1] = data[j][k];
            }
            double z = 0.0; // initialize logit
            for (int k = 0; k < num_features; k++) {
                z += coefficients[k] * x[k];
            }
            double h = 1.0 / (1.0 + exp(-z)); // sigmoid function
            cost += y * log(h) + (1.0 - y) * log(1.0 - h); // compute cost
            for (int k = 0; k < num_features; k++) {
                gradient[k] += (h - y) * x[k]; // accumulate gradient
            }
        }
        cost /= -num_samples;
        for (int j = 0; j < num_features; j++) {
            coefficients[j] -= learning_rate * gradient[j] / num_samples; // update coefficients
        }
    }
    return coefficients;
}

vector < double > gradient_descent(vector < vector < double >> x, vector < double > y, vector < double > coefficients, int num_iterations, double learning_rate) {
    int n = x.size();
    int m = x[0].size();
    for (int i = 0; i < num_iterations; i++) {
        vector < double > y_hat(n, 0.0);
        for (int j = 0; j < n; j++) {
            double z = 0.0;
            for (int k = 0; k < m; k++) {
                z += coefficients[k] * x[j][k];
            }
            y_hat[j] = 1.0 / (1.0 + exp(-z));
        }
        vector < double > errors(n, 0.0);
        for (int j = 0; j < n; j++) {
            errors[j] = y_hat[j] - y[j];
        }
        vector < double > new_coefficients(m, 0.0);
        for (int j = 0; j < m; j++) {
            double gradient = 0.0;
            for (int k = 0; k < n; k++) {
                gradient += errors[k] * x[k][j];
            }
            new_coefficients[j] = coefficients[j] - learning_rate * gradient;
        }
        coefficients = new_coefficients;
    }
    return coefficients;
}

// function to make predictions on new data
vector < double > predict(vector < vector < double >> data, vector < double > coefficients) {
    int num_samples = data.size(); // number of samples
    int num_features = data[0].size(); // number of features (including the intercept)
    vector < double > predictions(num_samples, 0.0); // initialize predictions to 0
    for (int i = 0; i < num_samples; i++) {
        double z = 0.0; // initialize logit
        for (int j = 0; j < num_features; j++) {
            z += coefficients[j] * data[i][j];
        }
        double h = 1.0 / (1.0 + exp(-z)); // sigmoid function
        predictions[i] = round(h); // round to 0 or 1
    }
    return predictions;

}

// function to calculate accuracy
double accuracy(vector < double > predictions, vector < double > actuals) {
    int num_samples = predictions.size(); // number of samples
    int num_correct = 0; // initialize number of correct predictions to 0
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] == actuals[i]) {
            num_correct++;
        }
    }
    return static_cast < double > (num_correct) / num_samples; // return the proportion of correct predictions
}

// function to calculate sensitivity
double sensitivity(vector < double > predictions, vector < double > targets) {
    int true_positives = 0;
    int false_negatives = 0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == 1.0 && targets[i] == 1.0) {
            true_positives++;
        } else if (predictions[i] == 0.0 && targets[i] == 1.0) {
            false_negatives++;
        }
    }
    if (true_positives == 0 && false_negatives == 0) {
        return 1.0; // all targets are negative, so sensitivity is undefined; return perfect sensitivity instead
    } else {
        return (double) true_positives / (true_positives + false_negatives);
    }
}

// function to calculate specificity
double specificity(vector < double > predictions, vector < double > targets) {
    int true_negatives = 0;
    int false_positives = 0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == 0.0 && targets[i] == 0.0) {
            true_negatives++;
        } else if (predictions[i] == 1.0 && targets[i] == 0.0) {
            false_positives++;
        }
    }
    if (true_negatives == 0 && false_positives == 0) {
        return 1.0; // all predictions are positive, so specificity is undefined; return perfect specificity instead
    } else {
        return (double) true_negatives / (true_negatives + false_positives);
    }
}

int main() {
    // load data
    vector < vector < double >> data = read_csv("titanic_project.csv");

    // separate data into train and test sets
    int train_size = 800;
    vector < vector < double >> train_data(data.begin(), data.begin() + train_size);
    vector < vector < double >> test_data(data.begin() + train_size, data.end());

    // extract target variable and predictor of interest
    vector < double > train_y(train_size, 0.0);
    vector < double > train_x(train_size, 0.0);
    for (int i = 0; i < train_size; i++) {
        train_y[i] = train_data[i][0]; // target variable
        train_x[i] = train_data[i][3]; // predictor of interest (sex)
    }

    // perform logistic regression
    int num_iterations = 1000;
    double learning_rate = 0.01;
    auto start_time = high_resolution_clock::now(); // start measuring time
    vector < vector < double >> train_data_x(train_size, vector < double > (2, 0.0)); // add intercept to train_x
    for (int i = 0; i < train_size; i++) {
        train_data_x[i][0] = 1.0; // intercept
        train_data_x[i][1] = train_x[i]; // predictor of interest (sex)
    }
    vector < double > initial_coefficients = {
        0.0,
        0.0
    }; // initialize coefficients
    vector < double > coefficients = gradient_descent(train_data_x, train_y, initial_coefficients, num_iterations, learning_rate); // get coefficients
    auto stop_time = high_resolution_clock::now(); // stop measuring time
    auto training_time = duration_cast < microseconds > (stop_time - start_time).count(); // get training time in microseconds
    cout << "Training time: " << training_time << " microseconds" << endl;
    cout << "Coefficients: " << coefficients[0] << " " << coefficients[1] << endl;

    // make predictions on test data
    vector < double > test_y(test_data.size(), 0.0);
    vector < double > test_x(test_data.size(), 0.0);
    for (int i = 0; i < test_data.size(); i++) {
        test_y[i] = test_data[i][0]; // target variable
        test_x[i] = test_data[i][3]; // predictor of interest (sex)
    }
    vector < vector < double >> test_data_x(test_data.size(), vector < double > (2, 0.0)); // add intercept to test_x
    for (int i = 0; i < test_data.size(); i++) {
        test_data_x[i][0] = 1.0; // intercept
        test_data_x[i][1] = test_x[i]; // predictor of interest (sex)
    }
    vector < double > test_predictions = predict(test_data_x, coefficients); // get test predictions

    // calculate metrics on test data
    double test_accuracy = accuracy(test_predictions, test_y);
    double test_sensitivity = sensitivity(test_predictions, test_y);
    double test_specificity = specificity(test_predictions, test_y);
    cout << "Test accuracy: " << test_accuracy << endl;
    cout << "Test sensitivity: " << test_sensitivity << endl;
    cout << "Test specificity: " << test_specificity << endl;

    return 0;
}