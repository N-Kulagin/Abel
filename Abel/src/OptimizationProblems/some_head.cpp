#include "pch.h"
#include "some_fold/some_head.h"

myStruct::myStruct(bool b, double d) : b(b), d(d){}

myStruct foo(int i, double d, bool b) {
	myStruct s(b, d);
	return s;
}
