#include "preprocessor.hpp"
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include "global_config.hpp"
#include "scip_solver.hpp"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void exit_and_help(){
	std::cout << "Usage: ./MIS [options] datafile" << std::endl;
	std::cout << "options: " << std::endl;

    std::cout << "-p : set the problem type" << std::endl;
    std::cout << "-0 - mis problem" << std::endl;
    std::cout << "-1 - sc problem" << std::endl;
    std::cout << "-2 - tsp problem" << std::endl;
	std::cout << "-3 - vrp problem" << std::endl;
	std::cout << "-4 - vc problem" << std::endl;
	std::cout << "-5 - ds problem" << std::endl;
	std::cout << "-6 - ca problem" << std::endl;

    std::cout << "-d : set the test dataset split" << std::endl;
    std::cout << "total 3 splits: 0..2" << std::endl;


    std::cout << "-b : set branching method to solve the problem (default 0)" << std::endl;
    std::cout << "0 .. 6" << std::endl;


    exit(1);
}


int main(int argc, char* argv[]) {
    using namespace COML;

    double param_cutoff      = 1000; //cutoff time in seconds.
    int    param_prob_type = 0;
    int    branching_policy = 0;    
    int    split_id = 0;
    int    span = 10;

    // parse options (parameters)
	for(int i = 1; i < argc; ++i){
		if(argv[i][0] != '-'){
		    break;
		}
		if(++i >= argc){
		    exit_and_help();
		}
		switch(argv[i-1][1]){
            case 't': param_cutoff = std::stod(argv[i]); break;
            case 'h': branching_policy = std::atoi(argv[i]); break;
            case 'p': param_prob_type = std::atoi(argv[i]); break;
            case 'd': split_id = std::atoi(argv[i]); break;
			default:
				std::cout << "Unknown option: " << argv[i-1][1] << std::endl;
				exit_and_help();
		}
	}

    Policy p;
    switch (branching_policy)
    {
        case 0: p = Policy::SCIP_DEF; break;
        case 1: p = Policy::SCIP_AGG; break;
        case 2: p = Policy::ML_DING; break;
        case 3: p = Policy::ML_DFS_HEUR_SCORE1_GCN; break;
        case 4: p = Policy::ML_DFS_HEUR_SCORE2_GCN; break;
        case 5: p = Policy::ML_DFS_HEUR_SCORE1_LR; break;
        case 6: p = Policy::ML_DFS_HEUR_SCORE2_LR; break;
        case 7: p = Policy::SCIP_HEUR_FEASPUMP; break;
        case 8: p = Policy::SCIP_HEUR_RENS; break;    
        case 9: p = Policy::SCIP_HEUR_ALL_ROUNDING; break;    
        case 10: p = Policy::SCIP_HEUR_ALL_DIVING; break;     

        case 11: p = Policy::ML_DFS_EXACT_SCORE1_GCN; break;
        case 12: p = Policy::ML_DFS_EXACT_SCORE2_GCN; break;
        case 13: p = Policy::ML_DFS_EXACT_SCORE3_GCN; break;
    
        case 14: p = Policy::SCIP_DEF_PBDFS; break;

        default: throw std::runtime_error("unknown branching_policy\n");
    }

    gconf.init(p, param_prob_type, param_cutoff, split_id);

    // specify testing & training graphs; optimal solutions for the training graphs must be provided.
    std::vector<std::string> test_files;
    for (int i = 0; i < 30; i++){
        test_files.push_back("eval_large/" + std::to_string(i));
    }
        

    for (auto test_file : test_files)
    {   
        gconf.cur_prob = test_file;
        Preprocessor preprocessor_ptr;
        int nvars;

        if (branching_policy == 2 || branching_policy == 3 || branching_policy == 4 ||
            branching_policy == 5 || branching_policy == 6 || branching_policy == 11 ||
            branching_policy == 12 || branching_policy == 13 || branching_policy == 14)
        {

            preprocessor_ptr.load_prob_map_gcn(gconf.var_mapper, branching_policy);            

            // get ml scores for each node
            const std::vector<double>& predicted_real_values = preprocessor_ptr.predicted_real_value;
            nvars = predicted_real_values.size();
            double ml_scores1[nvars];
            double ml_scores2[nvars];
            
            for (auto i = 0u; i < nvars; ++i){
                ml_scores1[i] = predicted_real_values[i];                       // (scoring fn: p)
                ml_scores2[i] = predicted_real_values[i];                       // (scoring fn: max(p, 1-p))
                if (ml_scores2[i] < 0.5) ml_scores2[i] = 1 - ml_scores2[i];
            }
            
            gconf.setup(nvars, ml_scores1, ml_scores2, 0);
        }else
            gconf.setup(0, NULL, NULL, 0);

        COML::SCIP_Solver solver;
        solver.solve();
        double obj = solver.finalize();
        gconf.cleanup();
    }
    
    return 0;
}
