#include "utils.hpp"
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <stdlib.h>
#include <random>
#include "global_config.hpp"
#include <limits>
#include <unordered_set>
#include <sstream>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string COML::currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M", &tstruct);

    return buf;
}




int COML::list_filenames_under_dir(std::string dir_name, std::vector<int> *vec)
{

    int max = -1;
    if (auto dir = opendir(dir_name.c_str()))
    {
        while (auto f = readdir(dir))
        {
            if (f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot

            int data_idx = std::stoi(f->d_name);
            if (vec != NULL)
                vec->push_back(data_idx);
            if (max < data_idx)
                max = data_idx;
        }
        closedir(dir);
        return max;
    }
    else
    {
        printf("in utils.list_filenames_under_dir(), open dir %s failed\n", dir_name.c_str());
        exit(EXIT_FAILURE);
    }
}

void COML::create_dir(std::string path)
{
    struct stat sb;
    if (!(stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
        mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

int COML::calc_argmax(const std::vector<double> &v)
{

    double max_score = -std::numeric_limits<double>::infinity();
    int arg_max = 0;
    int vec_size = v.size();
    //set true label index
    double tmp_score;
    for (int i = 0; i < vec_size; i++)
    {
        tmp_score = v.at(i);
        if (max_score < tmp_score)
        {
            max_score = tmp_score;
            arg_max = i;
        }
    }

    return arg_max;
}

void COML::all_zeros(double *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }
}


void COML::all_zeros(int *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }
}

double *COML::calc_top10(const double *arr, const int arr_size)
{
    std::vector<double> sb_scores_cp;
    double *_top10 = new double[10];

    sb_scores_cp.assign(arr, arr + arr_size);
    std::sort(sb_scores_cp.begin(), sb_scores_cp.end(), std::greater<double>());
    sb_scores_cp.resize(10);
    for (int i = 0; i < 10; i++)
    {
        _top10[i] = sb_scores_cp[i];
    }
    return _top10;
}

double *COML::calc_top1_5_10_30(const double *arr, const int arr_size)
{
#ifdef DEBUG
    printf("start calc_top1_5_10_30\n");
    fflush(stdout);
#endif

    std::vector<double> sb_scores_vec(arr, arr + arr_size);
    double *_top_1_5_10_30 = new double[4];

#ifdef DEBUG
    printf("after assign arr\n");
    fflush(stdout);
#endif

    std::sort(sb_scores_vec.begin(), sb_scores_vec.end(), std::greater<double>());

#ifdef DEBUG
    printf("after sorting\n");
    fflush(stdout);
#endif

    _top_1_5_10_30[0] = sb_scores_vec[0];
#ifdef DEBUG
    printf("after first assignment\n");
    fflush(stdout);
#endif

    if (arr_size >= 5)
        _top_1_5_10_30[1] = sb_scores_vec[4];
    else
        _top_1_5_10_30[1] = sb_scores_vec[arr_size - 1];
#ifdef DEBUG
    printf("after second assignment\n");
    fflush(stdout);
#endif
    if (arr_size >= 10)
        _top_1_5_10_30[2] = sb_scores_vec[9];
    else
        _top_1_5_10_30[2] = sb_scores_vec[arr_size - 1];
#ifdef DEBUG
    printf("after third assignment\n");
    fflush(stdout);
#endif
    if (arr_size >= 30)
        _top_1_5_10_30[3] = sb_scores_vec[29];
    else
        _top_1_5_10_30[3] = sb_scores_vec[arr_size - 1];

#ifdef DEBUG
    printf("done calc_top1_5_10_30\n");
    fflush(stdout);
#endif

    return _top_1_5_10_30;
}

int COML::calc_argmax(const double *arr, const int size)
{

    double max_score = -std::numeric_limits<double>::infinity();
    int arg_max = 0;

    for (int i = 0; i < size; i++)
    {
        if (max_score < arr[i])
        {
            max_score = arr[i];
            arg_max = i;
        }
    }
    return arg_max;
}

int COML::calc_argmin(const double *arr, const int size)
{

    double min_score = std::numeric_limits<double>::infinity();
    int arg_min = 0;

    for (int i = 0; i < size; i++)
    {
        if (min_score > arr[i])
        {
            min_score = arr[i];
            arg_min = i;
        }
    }
    return arg_min;
}

double COML::calc_max(const double *arr, const int arr_size)
{
    double maxval = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < arr_size; i++)
    {
        if (arr[i] > maxval)
        {
            maxval = arr[i];
        }
    }
    return maxval;
}

double COML::calc_min(const double *arr, const int arr_size)
{
    double minval = std::numeric_limits<double>::infinity();
    for (int i = 0; i < arr_size; i++)
    {
        if (arr[i] < minval)
        {
            minval = arr[i];
        }
    }
    return minval;
}

double COML::calc_mean(const double *arr, const int arr_size)
{
    double tot = 0.;
    int i;
    for (i = 0; i < arr_size; i++)
    {
        if (arr[i] == -100000000000000000000.)
            break;
        tot += arr[i];
    }
    return tot / (i);
}

double COML::calc_stdev(const double *arr, const int arr_size)
{
    double mean = COML::calc_mean(arr, arr_size);
    double variance = 0.;
    int i;
    for (i = 0; i < arr_size; i++)
    {
        if (arr[i] == -100000000000000000000.)
            break;
        variance += (arr[i] - mean) * (arr[i] - mean);
    }
    variance /= (i);
    return std::sqrt(variance);
}

double COML::calc_zscore(const double val, const double mean, const double stddev)
{
    double tmp = (val - mean) / stddev;
    return tmp;
}

void COML::min_max_norm(double *arr, int arr_size)
{
    double min = COML::calc_min(arr, arr_size);
    double max = COML::calc_max(arr, arr_size);
    double delta = max - min;
    for (int i = 0; i < arr_size; i++)
    {
        arr[i] = (delta == 0) ? 0 : (arr[i] - min) / delta;
        if (arr[i] < -1 || arr[i] > 1)
        {
            printf("%f\n", arr[i]);
        }
    }
}

void COML::arraycopy(double *src, double *dest, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        dest[i] = src[i];
    }
}

int COML::read_problems(std::string prefix, std::string filename, 
                    std::vector<std::string> &problems, std::vector<double> &objs)
{
    // read problems from file
    int num_problems = 0;
    std::ifstream infile(filename);
    if (infile.is_open())
    {
        std::string prob_name;
        std::string objval;
        while (getline(infile, prob_name))
        {   
            if (!prob_name.empty() && prob_name[prob_name.length() - 1] == '\n')
                prob_name.erase(prob_name.length() - 1);
            
            getline(infile, objval);
            if (!objval.empty() && objval[objval.length() - 1] == '\n')
                objval.erase(objval.length() - 1);

            problems.push_back(prefix + prob_name);
            objs.push_back(std::stod(objval));
            num_problems++;
        }
        infile.close();
    }
    else
    {
        printf("cannot find problem file: %s\n", filename.c_str());
        assert(false);
    }

    printf("number of problems to solve: %d\n\n", num_problems);
    return num_problems;
}

bool comparator_acending(const COML::mypair &l, const COML::mypair &r)
{
    return l.second < r.second;
}

bool COML::comparator_decending(const mypair &l, const mypair &r)
{
    return l.second > r.second;
}

void COML::argsort(const double *scores, int *sortedargs, int size)
{
    std::vector<mypair> score_index;
    for (int i = 0; i < size; i++)
    {
        mypair p;
        p.first = i;
        p.second = scores[i];
        score_index.push_back(p);
    }

    std::sort(score_index.begin(), score_index.end(), comparator_decending);

    for (int i = 0; i < size; i++)
    {
        sortedargs[i] = score_index[i].first;
    }
}

bool COML::is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}
