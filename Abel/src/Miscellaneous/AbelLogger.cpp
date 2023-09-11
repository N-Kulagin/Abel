#include "pch.h"
#include "Miscellaneous/AbelLogger.h"

AbelLogger::AbelLogger() {}

AbelLogger::AbelLogger(int N) : N(N)
{
	v.reserve(N);
	for (size_t i = 0; i < N; i++)
	{
		v.push_back(std::vector<double>());
		v[i].reserve(50);
	}
}

AbelLogger::AbelLogger(const AbelLogger& lg) : N(lg.N), v(lg.v) {}

AbelLogger& AbelLogger::operator=(const AbelLogger& lg)
{
	v = lg.v;
	N = lg.N;
	return *this;
}

void AbelLogger::record(const double& x, int col_num)
{
	v[col_num].push_back(x);
}

void AbelLogger::print(const std::string& s, const std::initializer_list<std::string>& l) const {

	if (N == 0) return;

	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch());

	std::ofstream fout;
	fout.open(s + "_" + std::to_string(ms.count()) + ".txt");

	auto name = l.begin();

	if (fout.is_open()) {
		for (const auto& el : v) {
			if (el.size() == 0) break;
			fout << *name << '\n';
			for (const auto& el2 : el) {
				fout << std::scientific << el2 << '\n';
			}
			fout << '\n';
			++name;
		}
		fout.close();
	}
	else { throw AbelException(ABEL_EX_MSG_CANT_OPEN_FILE, ABEL_EX_CODE_CANT_OPEN_FILE); }

}
