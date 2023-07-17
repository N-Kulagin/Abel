#pragma once

struct myStruct {
	bool b;
	double d;
	myStruct(bool, double);
};

myStruct foo(int i, double d, bool b);