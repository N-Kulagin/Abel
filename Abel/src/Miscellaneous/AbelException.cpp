#include "pch.h"
#include "Miscellaneous/AbelException.h"

AbelException::AbelException(const char* msg, int err_code) : std::exception(msg), err_code(err_code) {}

int AbelException::code()
{
	return err_code;
}
