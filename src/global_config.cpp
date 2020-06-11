#include "global_config.hpp"
#include "utils.hpp"
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <scip/struct_scip.h>
#include <scip/struct_tree.h>
#include <scip/struct_set.h>
#include <scip/struct_mem.h>
#include <scip/set.h>
#include <scip/tree.h>
#include <scip/cons_linear.h>
COML::Global_config::Global_config()
    : solving_stats_logger(NULL), optimization_type(Optimization_Type::Maximization),
        seed(0), cutoff(0), split_id(0), objs(), time(), ml_scores1(), ml_scores2(),
        dual_bounds(), dual_time(), gaps(), gap_time(), prefix_size_remove(3)
{}

void COML::Global_config::init(Policy policy, int prob_type, double cutoff_time, int split_id)
{
    const char *homedir;
    if ((homedir = getenv("HOME")) == NULL) 
        homedir = getpwuid(getuid())->pw_dir;
    
    this->home_dir = homedir; 
    this->start_time_clock = currentDateTime();
    this->split_id = split_id;

    this->cutoff = cutoff_time;
    this->policy = policy;
    this->prob = prob_type;
    int _cutoff = cutoff_time;
    std::string pname;
    if (policy == Policy::SCIP_AGG) pname = "scip_agg";
    else if (policy == Policy::SCIP_DEF) pname = "scip_def";
    else if (policy == Policy::SCIP_DEF_PBDFS) pname = "scip_def_pbdfs";
    else if (policy == Policy::ML_DING) pname = "ml_ding";
    else if (policy == Policy::ML_DFS_HEUR_SCORE1_GCN)
    {
        if (this->cutoff <= 100)
            pname = "ml_dfs1_gcn_" + std::to_string(_cutoff);
        else
            pname = "ml_dfs1_gcn";
    }
    else if (policy == Policy::ML_DFS_HEUR_SCORE2_GCN){
        if (this->cutoff <= 100)
            pname = "ml_dfs2_gcn_" + std::to_string(_cutoff);
        else
            pname = "ml_dfs2_gcn";
    } 
    else if (policy == Policy::ML_DFS_HEUR_SCORE3_GCN){
        if (this->cutoff <= 100)
            pname = "ml_dfs3_gcn_" + std::to_string(_cutoff);
        else
            pname = "ml_dfs3_gcn";
    }     else if (policy == Policy::ML_DFS_EXACT_SCORE1_GCN)
    {
        if (this->cutoff <= 100)
            pname = "pb_dfs1_gcn_exact_" + std::to_string(_cutoff);
        else
            pname = "pb_dfs1_gcn_exact";
    }
    else if (policy == Policy::ML_DFS_EXACT_SCORE2_GCN){
        if (this->cutoff <= 100)
            pname = "pb_dfs2_gcn_exact_" + std::to_string(_cutoff);
        else
            pname = "pb_dfs2_gcn_exact";
    } 
    else if (policy == Policy::ML_DFS_EXACT_SCORE3_GCN){
        if (this->cutoff <= 100)
            pname = "pb_dfs3_gcn_exact_" + std::to_string(_cutoff);
        else
            pname = "pb_dfs3_gcn_exact";
    } 
    else if (policy == Policy::ML_DFS_HEUR_SCORE2_LR){
        if (this->cutoff <= 100)
            pname = "ml_dfs2_lr_" + std::to_string(_cutoff);
        else
            pname = "ml_dfs2_lr";
    } 
    else if (policy == Policy::SCIP_HEUR_ALL_ROUNDING) pname = "heur_all_rounding";
    else if (policy == Policy::SCIP_HEUR_ALL_DIVING) pname = "heur_all_diving";
    else if (policy == Policy::SCIP_HEUR_FEASPUMP) pname = "heur_feaspump";
    else if (policy == Policy::SCIP_HEUR_RENS) pname = "heur_rens";

    else throw std::runtime_error("unrecognized policy!\n");
    switch (prob_type)
    {
    case 0: this->prob_str = "mis"; this->optimization_type = Maximization; break;
    case 1: this->prob_str = "sc"; this->optimization_type = Minimization; break;
    case 2: this->prob_str = "tsp"; this->optimization_type = Minimization; break;
    case 3: this->prob_str = "vrp"; this->optimization_type = Minimization; break;
    case 4: this->prob_str = "vc"; this->optimization_type = Minimization; break;
    case 5: this->prob_str = "ds"; this->optimization_type = Minimization; break;
    case 6: this->prob_str = "ca"; this->optimization_type = Maximization; break;

    default: throw std::runtime_error("unrecognized problem type!\n");
    }

    this->DATA_BASE_DIR = (home_dir + "/storage1/instances/" + prob_str + "/");
    this->LOG_DIR = home_dir + "/storage/ret_optsol/" + prob_str + "/" + pname + "/"; 
    COML::create_dir(home_dir + "/storage/ret_optsol/");
    COML::create_dir(home_dir + "/storage/ret_optsol/" + prob_str);
    COML::create_dir(this->LOG_DIR);

    std::string info(
        "problem type: " + prob_str + "\nevaluation policy: " + pname + 
        "\ndata read from: " + DATA_BASE_DIR + "\nlog write to: " + this->LOG_DIR + "\n\n");
    printf("%s\n", info.c_str());

    std::string tmp;
    tmp = LOG_DIR + "solving_stats_" + std::to_string(this->split_id) + ".csv";
    this->solving_stats_logger = new std::ofstream(tmp);
    (*solving_stats_logger) << "instance_id,status,opt_gap,best_sol_obj,best_sol_time,best_sol_heur,best_heur_sol_obj,best_heur_sol_time,heur_ncalls,heur_tot_time\n";
    (*solving_stats_logger).flush();

}


int comparator_decending1(const std::pair<std::string, double> &l, const std::pair<std::string, double> &r)
{
    return l.second > r.second;
}

void COML::Global_config::cleanup()
{
    objs.clear();
    time.clear();
    dual_bounds.clear();
    dual_time.clear();
    gaps.clear();
    gap_time.clear();
    first_sol_heur = "none";
    this->ml_scores1.clear();
    this->ml_scores2.clear();
    this->var_mapper.clear();
}

void COML::Global_config::setup(std::string prob_name, int nvars, double* ml_scores1,  double* ml_scores2, int seed)
{
    this->cur_prob = prob_name;
    this->seed = seed;
    this->nprobvars = nvars;
    for (int i = 0; i < nprobvars; i++){
        this->ml_scores1.push_back(ml_scores1[i]);
        this->ml_scores2.push_back(ml_scores2[i]);
    }
}



COML::Global_config::~Global_config()
{
    if (solving_stats_logger != NULL)
    {
        solving_stats_logger->flush();
        solving_stats_logger->close();
        delete solving_stats_logger;
    }
}


SCIP_RETCODE COML::Global_config::create_children_by_ml_local_cuts(SCIP* scip, 
                                SCIP_NODE** left_child_ptr, SCIP_NODE** right_child_ptr, bool left_child_only)
{

    int nfixed_vars;
    int rhs;
    switch (this->prob)
    {
    case 0: // mis prob
    case 4: // vc prob
    case 6:
        nfixed_vars = this->nprobvars * 0.9;
        rhs = 10;
        break;
    case 1: // sc
    case 5: // ds
        nfixed_vars = this->nprobvars * 0.9;
        rhs = 0;
        break;
    default:
        throw std::runtime_error("problem not specify fixing vars\n");
        break;
    }

    /* create mapping between var names and var */
    SCIP_VAR* var; SCIP_VAR* var1; SCIP_VAR* var2;
    int i; int j; int var_idx;
    std::string var_name; std::string var_name1; std::string var_name2; std::string prefix;
    std::vector<std::string> result;
    SCIP_VAR** vars = SCIPgetVars(scip); 
    int nvars = SCIPgetNVars(scip);
    printf("fixing variables: %d/%d\n", nfixed_vars, nvars);

    std::unordered_map<std::string, SCIP_VAR*> dict;
    for (i = 0; i < nvars; i++){
        dict[SCIPvarGetName(vars[i])] = vars[i];
    }

    // create var prefix
    var_name = SCIPvarGetName(vars[0]);
    prefix = var_name.substr(0, this->prefix_size_remove);
    
    SCIP_VAR* fixed_vars[nfixed_vars];
    double fixed_var_coefs[nfixed_vars];
    
    int subtract_to_rhs = 0;
    std::vector<std::pair<std::string, double>> name_score;

    for (i = 0; i < this->ml_scores2.size(); i++)
    {
        std::pair<std::string, double> p;
        p.first = this->orderednames[i];
        p.second = this->ml_scores2[i];
        name_score.push_back(p);
    }
    std::sort(name_score.begin(), name_score.end(), comparator_decending1);

    for (i = 0, j = 0; j < nfixed_vars; i++)
    {
        var_name = name_score[i].first;
        var_idx = this->var_mapper[var_name];
        if (dict.find( prefix + var_name ) != dict.end()){
            var = dict[prefix + var_name];
            var_idx = this->var_mapper[var_name];
            fixed_vars[j] = var;
            if (this->ml_scores1[var_idx] > 0.5)
            {
                subtract_to_rhs+=1;
                fixed_var_coefs[j] = - 1.;
            }
            else
            {
                fixed_var_coefs[j] = 1.;
            }
            j++;
        }
    }

    rhs = rhs - subtract_to_rhs;  

    printf("creating children by ml split!\n");
    
    SCIP_NODE* focusnode = SCIPgetFocusNode(scip);
    /* left */
    SCIP_NODE* left_child; SCIP_CONS* cons_left;

    SCIP_CALL( SCIPnodeCreateChild(&left_child, scip->mem->probmem, scip->set, scip->stat, scip->tree, 1, SCIPnodeGetEstimate(focusnode)));
    SCIPcreateConsLinear(scip, &cons_left, "cons_left_cut", nfixed_vars, fixed_vars, fixed_var_coefs, 
                    - SCIPsetInfinity(scip->set), rhs, 
                    TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE);
    SCIPaddConsNode(scip, left_child, cons_left, NULL);
    SCIPreleaseCons(scip, &cons_left); 
    *left_child_ptr = left_child;

    // left child should be the new focus subtree root
    if (left_child_only)
        return SCIP_OKAY;
    /* right */

    SCIP_NODE* right_child; SCIP_CONS* cons_right;
    SCIP_CALL( SCIPnodeCreateChild(&right_child, scip->mem->probmem, scip->set, scip->stat, scip->tree, 0, SCIPnodeGetEstimate(focusnode)));

    SCIPcreateConsLinear(scip, &cons_right, "cons_left_cut", nfixed_vars, fixed_vars, fixed_var_coefs,
                    rhs+1 , SCIPsetInfinity(scip->set), 
                    TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE);
    SCIPaddConsNode(scip, right_child, cons_right, NULL);
    SCIPreleaseCons(scip, &cons_right); 
    *right_child_ptr = right_child;
}

COML::Global_config gconf;


