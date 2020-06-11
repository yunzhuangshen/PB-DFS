


#ifndef _MIPEXP_UTILS
#define _MIPEXP_UTILS

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <map> 
#include <scip/scip.h>
#include <string>
#include <algorithm>
#include <cassert>
#include <fstream>
#include "global_config.hpp"
#include <time.h>
namespace COML
{   
    const std::string currentDateTime();
    typedef std::pair<int,double> mypair;
    bool comparator_decending ( const mypair& l, const mypair& r);
    int list_filenames_under_dir(std::string dir_name, std::vector<int>* vec);
    void create_dir(std::string path);
    int read_problems(std::string prefix, std::string filename, 
                    std::vector<std::string>& problems, std::vector<double> &objs);
    int calc_argmax(const double* arr, const int size );
    int calc_argmax(const std::vector<double>& v );
    int calc_argmin(const double* arr, const int size );
    double* calc_top10(const double* arr, const int arr_size);
    double* calc_top1_5_10_30(const double* arr, const int arr_size);
    double calc_max(const double* arr, const int arr_size);
    double calc_min(const double* arr, const int arr_size);
    double calc_mean(const double* arr, const int arr_size);
    double calc_stdev(const double* arr, const int arr_size);
    double calc_zscore(const double val, const double mean, const double stddev);
    void min_max_norm(double * arr, const int arr_size);
    void arraycopy(double* src, double* dest, int num_elements);
    void argsort(const double* scores, int* sortedargs, int size);
    void all_zeros(int* arr, int size);
    void all_zeros(double* arr, int size);

    bool is_file_exist(const char *fileName);
}

#endif