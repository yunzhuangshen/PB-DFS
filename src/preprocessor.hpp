#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H


#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <numeric>      // std::iota
#include <vector>
#include <cstring>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <unordered_map>

namespace COML {
class Preprocessor
{
public:
    std::vector<double> predicted_real_value;
    explicit  Preprocessor();
    void load_prob_map_gcn(std::unordered_map<std::string, int>& mapper, 
                        int branching_policy );
};
}

#endif
