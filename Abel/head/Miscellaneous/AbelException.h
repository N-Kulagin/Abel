#pragma once

class AbelException : public std::exception {

public:
	AbelException(const char* msg, int err_code);
	int code();
private:
	int err_code;
};