#include "preprocessor.hpp"
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include "global_config.hpp"
#include "scip_solver.hpp"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"

void solve(std::string path)
{

}

int main0(int argc, char* argv[]) {
    using namespace COML;

    const char * _homedir;
    if ((_homedir = getenv("HOME")) == NULL) 
        _homedir = getpwuid(getuid())->pw_dir;
    
    std::string home_dir = _homedir;
    std::string prob = "ca";
    std::string data_dir = home_dir + "/storage1/instances/" + prob + "/";
    std::string lp_path;

    
    double nnzs;
    int nvars;
    int ncons;
    double frac;

    int nvars_max;
    int nvars_min;
    int ncons_max;
    int ncons_min;
    double frac_max;
    double frac_min;

    // mis problem
    nvars_max = -1;
    nvars_min = 1000000000;
    ncons_max = -1;
    ncons_min = 1000000000;
    frac_max = 0;
    frac_min = 1;

    SCIP* scip;
    for (int i = 0; i < 30; i++){
        // lp_path = data_dir + "test_2000/" + std::to_string(i) + ".lp";
        lp_path = data_dir + "eval_300-1500/" + std::to_string(i) + ".lp";

        SCIPcreate(&scip);
        assert(scip!=NULL);
        SCIPincludeDefaultPlugins(scip);
        printf("%s\n", lp_path.c_str());
        SCIPreadProb(scip, lp_path.c_str(), NULL);
        SCIPsetIntParam(scip, "presolving/maxrestarts", 0);
        SCIPsetIntParam(scip, "separating/maxrounds", 0);
        SCIPsetIntParam(scip, "separating/maxroundsroot", 0);
        SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip), TRUE);
        SCIPtransformProb(scip);
        // SCIPsolve(scip);
        nvars = SCIPgetNOrigBinVars(scip);
        ncons = SCIPgetNOrigConss(scip);
        nnzs = SCIPgetNNZs(scip);
        frac = nnzs / nvars / ncons; 
        SCIPfree(&scip);

        if (nvars > nvars_max)
            nvars_max = nvars;
        if (nvars < nvars_min)
            nvars_min = nvars;
        if (ncons > ncons_max)
            ncons_max = ncons;
        if (ncons < ncons_min)
            ncons_min = ncons;
        if (frac > frac_max)
            frac_max = frac;
        if (frac < frac_min)
            frac_min = frac;
    }
    printf("mis large: %d-%d %d-%d %f-%f\n", nvars_min, nvars_max, ncons_min, ncons_max, frac_min, frac_max);

    // specify testing & training graphs; optimal solutions for the training graphs must be provided.
    // std::vector<std::string> train_files; std::vector<std::string> test_files;
    // switch (gconf.prob)
    // {
    // case 0: //MIS problem

    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_3000/" + std::to_string(i));
    //     }
      
    //     break;
    // case 1: // Setcover problem
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_2250_1750/" + std::to_string(i));
    //     }
    //     break;

    // case 2: // TSP
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_150/" + std::to_string(i));
    //     } 
    //     break;
    // case 3: // vrp
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_50/" + std::to_string(i));
    //     }
    //     break;
    // case 4: // vc
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_3000/" + std::to_string(i));
    //     }
    //     break;
    // case 5: // dominate set
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_3000/" + std::to_string(i));
    //     }
    //     break;
    // case 6: // ca
    //     for (int i = 0; i < 30; i++){
    //         test_files.push_back("eval_300-1500/" + std::to_string(i));
    //     }
    //     break;
    // default: throw std::runtime_error("unknown problem\n");
    // }


    return 0;
}
