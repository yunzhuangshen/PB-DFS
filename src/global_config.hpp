#ifndef __global__conf__
#define __global__conf__

#include <cstdlib>
#include <string>
#include <pwd.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <boost/algorithm/string.hpp>
#include <scip/type_retcode.h>
#include <scip/scip.h>

namespace COML{
typedef enum Optimization_Type
{
    Maximization, Minimization
} Optimization_Type;

typedef enum Policy
{
    // exact
    SCIP_DEF_PBDFS,
    SCIP_DEF,
    SCIP_AGG,
    ML_DING,
    ML_DFS_HEUR_SCORE1_GCN,
    ML_DFS_HEUR_SCORE2_GCN,
    ML_DFS_HEUR_SCORE1_LR,
    ML_DFS_HEUR_SCORE2_LR,
    SCIP_HEUR_FEASPUMP,
    SCIP_HEUR_RENS,
    SCIP_HEUR_ALL_ROUNDING,
    SCIP_HEUR_ALL_DIVING,
    ML_DFS_EXACT_SCORE1_GCN,
    ML_DFS_EXACT_SCORE2_GCN,
    ML_DFS_EXACT_SCORE3_GCN,
    STATS,
} Policy;

typedef struct Global_config
{
    Global_config();
    ~Global_config();
public:

    std::string prob_str;
    Optimization_Type optimization_type;
    int split_id;
    std::string prob_lp_path;

    int prefix_size_remove;
    int nprobvars;
    int prob;
    std::unordered_map<std::string, int> var_mapper;
    std::vector<std::string> orderednames;
    std::vector<double> ml_scores1; // fn = p
    std::vector<double> ml_scores2; // fn = max(1-p, p)

    Policy policy;  
    std::string cur_prob;
    double cutoff;
    std::string start_time_clock;
    std::string home_dir;
    std::string DATA_BASE_DIR;
    std::string LOG_DIR;
    std::ofstream* solving_stats_logger;
    bool record_solving_info = true;

    std::string first_sol_heur;

    // recorded information during solving
    std::vector<double> objs;
    std::vector<double> time;
    std::vector<double> dual_bounds;
    std::vector<double> dual_time;
    std::vector<double> gaps;
    std::vector<double> gap_time;
    void init(Policy policy, int prob_type, double cutoff_time, int split_id);
    void setup(std::string prob_name, int nnodes, double* ml_scores1, double* ml_scores2, int seed);
    void cleanup();    // must be used at the end 
    SCIP_RETCODE create_children_by_ml_local_cuts(SCIP* scip, 
                                SCIP_NODE** left_child_ptr, SCIP_NODE** right_child_ptr, bool left_child_only);

    // --- scip param variables ---
    int seed;
    
}Global_config;
}

extern COML::Global_config gconf;

#endif