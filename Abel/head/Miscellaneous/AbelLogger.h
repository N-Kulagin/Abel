#pragma once

// class used for keeping track of numerical values of algorithms
class AbelLogger {

private:
	std::vector<std::vector<double>> v; // store numerical values in this (jagged) array
	int N = 0; // number of variables to store

public:
	AbelLogger();
	AbelLogger(int N);
	AbelLogger(const AbelLogger& lg);
	AbelLogger& operator=(const AbelLogger& lg);
	void record(const double& x, int col_num);

	void print(const std::string& s, const std::initializer_list<std::string>& l) const;
};
