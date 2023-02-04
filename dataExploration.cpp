#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

// a function to find the sum of a numeric vector
double getSum(vector<double> v) {
    double sum = 0;
    for (int i=0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum;
}

// a function to find the mean of a numeric vector
double getMean(vector<double> v) {\
    double sum = getSum(v);
    double size = v.size();
    return sum/size;
}

// a function to find the median of a numeric vector
double getMedian(vector<double> v) {
    int size = v.size();

    // using the built-in sort() function
    sort(v.begin(), v.end());

    // case I: Vector size is even. Average the middle two elements
    if (size % 2 == 0) {
        return (v[size / 2 - 1] + v[size / 2]) / 2;
    }// case II: Vector size is odd. return the middle element
    else {
        return v[size / 2];
    }
}

// a function to find the range of a numeric vector
double getRange(vector<double> v) {
    double max = INT_MIN;
    double min = INT_MAX;

    for (int i=0; i < v.size(); i++) {
        // check for max
        if (v[i] > max) {
            max = v[i];
        }
        // check for min
        if (v[i] < min) {
            min = v[i];
        }
    }
    return max - min;
}

// a function to compute covariance between rm and medv
double covar(vector<double> rmData, vector<double> medvData) {
    double rmMean = getMean(rmData);
    double medvMean = getMean(medvData);

    double div = 0.0;
    for (int i=0; i <rmData.size(); i++) {
        div += (rmData[i] - rmMean) * (medvData[i] - medvMean);
    }

    return div / (rmData.size() - 1);

}

// a function to compute correlation between rm and medv
double cor(vector<double> rmData, vector<double> medvData) {
    // calculating the sigma values for the rm and medv dataset
    double rmSig = sqrt(covar(rmData, rmData));
    double medvSig = sqrt(covar(medvData, medvData));

    return covar(rmData, medvData) / (rmSig * medvSig);
}

void printStats(vector<double> data){
    cout << "Sum = " << getSum(data) << endl;
    cout << "Mean = " << getMean(data) << endl;
    cout << "Median = " << getMedian(data) << endl;
    cout << "Range = " << getRange(data) << endl;
}

int main() {
    vector<double> rm;
    vector<double> medv;

    // opening file
    cout << "Opening file Boston.csv..." << endl;
    ifstream inFile("Boston.csv");

    // exit with code 1 indicating error
    if (!inFile.is_open()) {
        cout << "Failed to open file." << endl;
        return 1;
    }
    cout << "File opened!" << endl;
    
    string line;
    getline(inFile, line);

    // read in the numbers
    string rmNum;
    string medvNum;

    while (getline(inFile, line)) {
        stringstream ss(line);
        getline(ss, rmNum, ',');
        getline(ss, medvNum);

        // first value goes to rm
        rm.push_back(stof(rmNum));
        // second value goes to medv
        medv.push_back(stof(medvNum));
    }

    // close the file
    cout << "Closing file Boston.csv." << endl << endl;
    inFile.close();

    // print out the stats on the datasets
    cout << "Stats for rm:" << endl;
    printStats(rm);
    cout << endl;

    cout << "Stats for medv:" << endl;
    printStats(medv);
    cout << endl;

    // print out their covariance and correlation
    cout << "The covariance between rm and medv is " << covar(rm, medv) << endl;
    cout << "The correlation between rm and medv is " << cor(rm, medv) << endl;
}
