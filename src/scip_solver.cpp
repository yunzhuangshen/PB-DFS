#include <string>
#include <algorithm>
#include <utility>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <scip/debug.h>
#include <scip/struct_scip.h>
#include <scip/type_scip.h>
#include <scip/type_primal.h>
#include <scip/struct_primal.h>
#include <scip/type_stat.h>
#include <scip/struct_stat.h>
#include <scip/set.h>
#include <scip/struct_sol.h>
#include <scip/struct_misc.h>
#include <scip/type_sol.h>
#include <scip/type_misc.h>
#include <scip/type_retcode.h>
#include <scip/pub_message.h>
#include <boost/algorithm/string.hpp> 
#include "scip_solver.hpp"
#include "scip_exception.hpp"
#include "global_config.hpp"
#include "utils.hpp"
#include "heur_ml_diving.hpp"
#include "branching_policy.hpp"
#include "heur_ml_subscip.hpp"

static SCIP_RETCODE includeSCIPotherPlugins(
   SCIP*                 scip                /**< SCIP data structure */
);
static SCIP_RETCODE includeSCIPheuristicsDoRequireFeasibleSol(
   SCIP*                 scip                /**< SCIP data structure */
);

static SCIP_RETCODE includeSCIPheuristicsDoNotRequireFeasibleSol(
   SCIP*                 scip                /**< SCIP data structure */
);

// for data collection
COML::SCIP_Solver::SCIP_Solver()
    : scip(NULL), nodesel_policy(NULL), branching_policy(NULL)
{
   std::string prob_name = gconf.DATA_BASE_DIR + gconf.cur_prob + ".lp";
   gconf.prob_lp_path = prob_name;
   
   // initialize scip
   SCIP_CALL_EXC(SCIPcreate(&scip));

   assert(scip!=NULL);
   
   branching_policy = new COML::Branching(
      scip, "db", "", 1000000, -1, 1);
   COML::SCIPincludeObjBranchrule_DB(scip, branching_policy, false);
   includeSCIPotherPlugins(scip);

   if (gconf.policy == Policy::ML_DFS_HEUR_SCORE1_GCN ||
               gconf.policy == Policy::ML_DFS_HEUR_SCORE2_GCN || 
               gconf.policy == Policy::ML_DFS_HEUR_SCORE1_LR ||
               gconf.policy == Policy::ML_DFS_HEUR_SCORE2_LR)
   {
      SCIPincludeHeurMLdfs(scip);
   }else if (gconf.policy == Policy::ML_DFS_EXACT_SCORE1_GCN ||
               gconf.policy == Policy::ML_DFS_EXACT_SCORE2_GCN || 
               gconf.policy == Policy::ML_DFS_EXACT_SCORE3_GCN)
   {
      nodesel_policy = new COML::Nodesel_ML_DFS(scip, "", "", 66666667, 66666667);
      SCIPincludeObjNodesel_MLDFS(scip, nodesel_policy, false);
      SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE);
   }
   else if (gconf.policy == Policy::SCIP_AGG ||
               gconf.policy == Policy::SCIP_DEF)
   {
      includeSCIPheuristicsDoNotRequireFeasibleSol(scip);
      includeSCIPheuristicsDoRequireFeasibleSol(scip);

      if (gconf.policy == Policy::SCIP_AGG)
      {
         SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE);
      }
   }
   else if (gconf.policy == Policy::SCIP_DEF_PBDFS)
   {
      includeSCIPheuristicsDoNotRequireFeasibleSol(scip);
      includeSCIPheuristicsDoRequireFeasibleSol(scip);
      SCIPincludeHeurMLdfs(scip);
   }
   else
   {  
      if (gconf.policy == Policy::SCIP_HEUR_ALL_DIVING)
      {
         SCIPincludeHeurCoefdiving(scip);
         SCIPincludeHeurDistributiondiving(scip);
         SCIPincludeHeurFracdiving(scip);
         SCIPincludeHeurIntdiving(scip);
         SCIPincludeHeurLinesearchdiving(scip);
         SCIPincludeHeurNlpdiving(scip);
         SCIPincludeHeurObjpscostdiving(scip);
         SCIPincludeHeurPscostdiving(scip);
         SCIPincludeHeurRootsoldiving(scip);
         SCIPincludeHeurVeclendiving(scip);
         SCIPincludeHeurOctane(scip);
         SCIPincludeHeurConflictdiving(scip);
         SCIPincludeHeurFarkasdiving(scip);
         SCIPincludeHeurActconsdiving(scip);
         SCIPincludeHeurGuideddiving(scip);
      }
      else if (gconf.policy == Policy::SCIP_HEUR_ALL_ROUNDING)
      {
         SCIPincludeHeurIntshifting(scip);
         SCIPincludeHeurRounding(scip);
         SCIPincludeHeurRandrounding(scip);
         SCIPincludeHeurSimplerounding(scip);
         SCIPincludeHeurRounding(scip);
         SCIPincludeHeurZirounding(scip);
         SCIPincludeHeurShiftandpropagate(scip);
         SCIPincludeHeurShifting(scip);
      }
      else if (gconf.policy == Policy::SCIP_HEUR_FEASPUMP)
         SCIPincludeHeurFeaspump(scip);     // SCIP_HEURTIMING_AFTERLPPLUNGE freq 20
      else if (gconf.policy == Policy::SCIP_HEUR_RENS)
         SCIPincludeHeurRens(scip);         // SCIP_HEURTIMING_AFTERLPNODE freq 0
      else if (gconf.policy != Policy::ML_DING) 
         throw std::runtime_error("unrecognized policy in scip_solver!\n");
   }

   // disable scip output to stdout
   SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip), TRUE);

   scip_set_params();

   printf("\nread problem from %s\n", prob_name.c_str());
   SCIPreadProb(scip, prob_name.c_str(), NULL);

}

/* destructor */
COML::SCIP_Solver::~SCIP_Solver(void)
{
   if (scip != NULL)
   {  
      SCIP_CALL_EXC(SCIPfree(&scip));
      scip = NULL;
   }

   if (branching_policy != NULL)
   {
      delete branching_policy;
      branching_policy = NULL;
   }

   if (nodesel_policy != NULL)
   {
      delete nodesel_policy;
      nodesel_policy = NULL;
   }
}


void COML::SCIP_Solver::scip_set_params()
{
   /** 
      some important scip parameters: -1 means unlimited, 0 means disabled
      ------------------------------------------------------------------------
    */
   // time limit
   SCIPsetRealParam(scip, "limits/time", gconf.cutoff);
   SCIPsetIntParam(scip, "display/verblevel", 5);
   SCIPsetIntParam(scip, "timing/clocktype", 2);

   // random seed to break ties
   SCIPsetBoolParam(scip, "randomization/permutevars", TRUE);
   SCIPsetIntParam(scip, "randomization/permutationseed", gconf.seed);
   SCIPsetIntParam(scip, "randomization/randomseedshift", gconf.seed);

   SCIPsetIntParam(scip, "presolving/maxrestarts", 0);

   if (gconf.cutoff <= 100)
   {
      SCIPsetIntParam(scip, "separating/maxrounds", 0);
      SCIPsetIntParam(scip, "separating/maxroundsroot", 0);
   }

   // SCIPsetIntParam(scip, "presolving/maxrounds", 0);
}

/* display the solving statistics */
double COML::SCIP_Solver::finalize()
{
   
  printf("nnodes left in finalize: %d\n", SCIPgetNNodesLeft(scip));
   double solving_time = std::round(SCIPgetSolvingTime(scip));
   long long num_nodes = SCIPgetNNodes(scip);
   long long num_internal_nodes = scip->stat->ninternalnodes;
   long long num_lps_sb = SCIPgetNStrongbranchLPIterations(scip);
   long long num_lps_relax = SCIPgetNNodeLPIterations(scip);
   double dual_bound = SCIPgetDualbound(scip);
   //|(primalbound - dualbound)/min(|primalbound|,|dualbound|)|
   double opt_gap = abs((SCIPgetPrimalbound(scip) - SCIPgetDualbound(scip))/(SCIPgetPrimalbound(scip) + SCIPsetEpsilon(scip->set)));
   
   std::string status;
   SCIP_STATUS _status = SCIPgetStatus(scip);

   if (_status == SCIP_STATUS_OPTIMAL ||
       _status == SCIP_STATUS_INFEASIBLE ||
       _status == SCIP_STATUS_UNBOUNDED ||
       _status == SCIP_STATUS_INFORUNBD)
      status = "solved";
   else
      status = "unsolved";

   double best_sol_obj = -1;
   double best_sol_time = -1;


   /**
    * write primal bound plot
   */
   int i;
   std::vector<std::string> result; 
   boost::split(result, gconf.cur_prob, boost::is_any_of("/")); 
   COML::create_dir(gconf.LOG_DIR + result[0]);

   std::string tmp = gconf.LOG_DIR + gconf.cur_prob + ".primal_plot";
   std::ofstream fsol_logger(tmp);
   fsol_logger << gconf.objs.size() << "\n";
   printf("write primal curve to %s\n", tmp.c_str());
   for (i = 0; i < gconf.objs.size(); i++)
      fsol_logger << gconf.time[i] << " ";
   fsol_logger << "\n";
   for (i = 0; i < gconf.objs.size(); i++)
      fsol_logger << gconf.objs[i] << " ";
   fsol_logger.flush();
   fsol_logger.close();
   
   /**
    * write dual bound plot
   */

   tmp = gconf.LOG_DIR + gconf.cur_prob + ".dual_plot";
   printf("write dual curve to %s\n", tmp.c_str());
   std::ofstream fdual_logger(tmp);
   fdual_logger << gconf.dual_bounds.size() << "\n";
   for (i = 0; i < gconf.dual_bounds.size(); i++)
      fdual_logger << gconf.dual_time[i] << " ";
   fdual_logger << "\n";
   for (i = 0; i < gconf.dual_bounds.size(); i++)
      fdual_logger << gconf.dual_bounds[i] << " ";
   fdual_logger.flush();
   fdual_logger.close();

   /**
    * write gap plot
   */
   tmp = gconf.LOG_DIR + gconf.cur_prob + ".gap_plot";
   printf("write optgap curve to %s\n", tmp.c_str());
   std::ofstream gap_logger(tmp);
   gap_logger << gconf.gaps.size() << "\n";
   for (i = 0; i < gconf.gaps.size(); i++)
      gap_logger << gconf.gap_time[i] << " ";
   gap_logger << "\n";
   for (i = 0; i < gconf.gaps.size(); i++)
      gap_logger << gconf.gaps[i] << " ";
   gap_logger.flush();
   gap_logger.close();

      /**
    * heuristic stats
    */
   tmp = gconf.LOG_DIR + gconf.cur_prob + ".heur_stats";
   std::ofstream heur_logger(tmp);
   SCIP_HEUR** heuristics = SCIPgetHeurs(scip);
   int nheuristics = SCIPgetNHeurs(scip);
   SCIP_HEUR* heuristic = NULL;
   heur_logger << "name,nbestsol,nsol,ncalls,time\n";
   for (i = 0; i < nheuristics; i++)
   {
      heuristic = heuristics[i];
      heur_logger << SCIPheurGetName(heuristic) << ",";
      heur_logger << SCIPheurGetNBestSolsFound(heuristic) << ",";
      heur_logger << SCIPheurGetNSolsFound(heuristic) << ",";
      heur_logger << SCIPheurGetNCalls(heuristic) << ",";
      heur_logger << SCIPheurGetTime(heuristic) << "\n";
   }

   heur_logger.flush();
   heur_logger.close();


   SCIP_SOL* best_solution = SCIPgetBestSol(scip);
   SCIP_HEUR* best_sol_heur = NULL;
   std::string best_sol_heur_name;
   if (best_solution != NULL)
   {
      best_sol_obj = SCIPgetSolOrigObj(scip, best_solution);
      best_sol_time = SCIPgetSolTime(scip, best_solution);
      best_sol_heur = SCIPgetSolHeur(scip, best_solution);
      if (best_sol_heur != NULL)
         best_sol_heur_name = SCIPheurGetName(best_sol_heur);
      else
         best_sol_heur_name = "none";
   }

   double first_sol_time = gconf.time.size() == 0 ? -1 : gconf.time[0];
   double first_sol_obj = gconf.objs.size() == 0 ? -1 : gconf.objs[0];

   printf("status:%s,opt_gap:%f,best_sol_obj:%f,best_sol_time:%f,best_sol_heur:%s,first_sol_obj:%f,first_sol_time:%f,first_sol_heur:%s,global_dual_bound:%f\n",
            status.c_str(), opt_gap, best_sol_obj, best_sol_time, best_sol_heur_name.c_str(), first_sol_obj, first_sol_time, gconf.first_sol_heur.c_str(), dual_bound);

   int totalcalls = 0;
   double heuristic_total_time = 0.;
   for (i = 0; i < nheuristics; i++)
   {
      heuristic = heuristics[i];
      totalcalls += SCIPheurGetNCalls(heuristic);
      heuristic_total_time += SCIPheurGetTime(heuristic);
   }

   int nsols = SCIPgetNSols(scip);
   SCIP_SOL** sols = SCIPgetSols(scip);
   SCIP_HEUR* h;

   double best_heur_sol_time = 0.;
   double best_heur_sol_obj = 0.;
   double obj;
   for (i = 0; i < nsols; i++)
   {
      h = SCIPsolGetHeur(sols[i]);
      obj = SCIPgetSolOrigObj(scip, sols[i]);
      if (h!=NULL)
      {  
         if (best_heur_sol_obj==0.)
         {  
            best_heur_sol_obj = obj;
            best_heur_sol_time = SCIPsolGetTime(sols[i]);
         }
         else if (
               (gconf.optimization_type == Optimization_Type::Maximization && 
                     obj > best_heur_sol_obj + SCIPsetEpsilon(scip->set)) ||
               (gconf.optimization_type == Optimization_Type::Minimization && 
                     obj < best_heur_sol_obj - SCIPsetEpsilon(scip->set))
         )
         {
            best_heur_sol_obj = obj;
            best_heur_sol_time = SCIPsolGetTime(sols[i]);
         }
      }
   }

   // instance_id,status,opt_gap,
   // best_sol_obj,best_sol_time,best_sol_heur,
   // best_heur_sol_obj,best_heur_sol_time,
   // heur_ncalls,heur_tot_time
   (*gconf.solving_stats_logger) << gconf.cur_prob  << "," << status << ","  << opt_gap << ","
                                 << best_sol_obj << "," << best_sol_time << "," << best_sol_heur_name << ","
                                 << best_heur_sol_obj << "," <<  best_heur_sol_time << ","
                                 << totalcalls << "," << heuristic_total_time << "\n";
   (*gconf.solving_stats_logger).flush();

   return best_sol_obj;
}

void COML::SCIP_Solver::solve(void)
{
   // this tells scip to start the solution process
   SCIP_CALL_EXC(SCIPsolve(scip));
}


static SCIP_RETCODE includeSCIPheuristicsDoNotRequireFeasibleSol(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   // heuristics do not require a feasible solution
   SCIP_CALL( SCIPincludeHeurCoefdiving(scip) );
   SCIP_CALL( SCIPincludeHeurDistributiondiving(scip) );
   SCIP_CALL( SCIPincludeHeurFracdiving(scip) );
   SCIP_CALL( SCIPincludeHeurIntdiving(scip) );
   SCIP_CALL( SCIPincludeHeurLinesearchdiving(scip) );
   SCIP_CALL( SCIPincludeHeurNlpdiving(scip) );
   SCIP_CALL( SCIPincludeHeurObjpscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurPscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurRootsoldiving(scip) );
   SCIP_CALL( SCIPincludeHeurVeclendiving(scip) );
   SCIP_CALL( SCIPincludeHeurOctane(scip) );
   SCIP_CALL( SCIPincludeHeurFeaspump(scip) );
   SCIP_CALL( SCIPincludeHeurRens(scip) );
   SCIP_CALL( SCIPincludeHeurIntshifting(scip) );
   SCIP_CALL( SCIPincludeHeurActconsdiving(scip) );
   SCIP_CALL( SCIPincludeHeurRandrounding(scip) );
   SCIP_CALL( SCIPincludeHeurSimplerounding(scip) );
   SCIP_CALL( SCIPincludeHeurRounding(scip) );
   SCIP_CALL( SCIPincludeHeurZirounding(scip) );
   SCIP_CALL( SCIPincludeHeurShiftandpropagate(scip) );
   SCIP_CALL( SCIPincludeHeurShifting(scip) );
   SCIP_CALL( SCIPincludeHeurConflictdiving(scip) );
   SCIP_CALL( SCIPincludeHeurFarkasdiving(scip) );
   
}


static SCIP_RETCODE includeSCIPheuristicsDoRequireFeasibleSol(
   SCIP*                 scip                /**< SCIP data structure */
)
{

   // heuristics do require a feasible solution
   
   SCIP_CALL( SCIPincludeHeurGuideddiving(scip) );
   SCIP_CALL( SCIPincludeHeurBound(scip) );
   SCIP_CALL( SCIPincludeHeurClique(scip) );
   SCIP_CALL( SCIPincludeHeurCompletesol(scip) );
   SCIP_CALL( SCIPincludeHeurCrossover(scip) );
   SCIP_CALL( SCIPincludeHeurDins(scip) );
   SCIP_CALL( SCIPincludeHeurDualval(scip) );
   SCIP_CALL( SCIPincludeHeurFixandinfer(scip) );
   SCIP_CALL( SCIPincludeHeurGins(scip) );
   SCIP_CALL( SCIPincludeHeurZeroobj(scip) );
   SCIP_CALL( SCIPincludeHeurIndicator(scip) );
   SCIP_CALL( SCIPincludeHeurLocalbranching(scip) );
   SCIP_CALL( SCIPincludeHeurLocks(scip) );
   SCIP_CALL( SCIPincludeHeurLpface(scip) );
   SCIP_CALL( SCIPincludeHeurAlns(scip) );
   SCIP_CALL( SCIPincludeHeurMutation(scip) );
   SCIP_CALL( SCIPincludeHeurMultistart(scip) );
   SCIP_CALL( SCIPincludeHeurMpec(scip) );
   SCIP_CALL( SCIPincludeHeurOfins(scip) );
   SCIP_CALL( SCIPincludeHeurOneopt(scip) );
   SCIP_CALL( SCIPincludeHeurProximity(scip) );
   SCIP_CALL( SCIPincludeHeurReoptsols(scip) );
   SCIP_CALL( SCIPincludeHeurRepair(scip) );
   SCIP_CALL( SCIPincludeHeurRins(scip) );
   SCIP_CALL( SCIPincludeHeurSubNlp(scip) );
   SCIP_CALL( SCIPincludeHeurTrivial(scip) );
   SCIP_CALL( SCIPincludeHeurTrivialnegation(scip) );
   SCIP_CALL( SCIPincludeHeurTrySol(scip) );
   SCIP_CALL( SCIPincludeHeurTwoopt(scip) );
   SCIP_CALL( SCIPincludeHeurUndercover(scip) );
   SCIP_CALL( SCIPincludeHeurVbounds(scip) );
}

static SCIP_RETCODE includeSCIPotherPlugins(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   SCIP_CALL( SCIPincludeConshdlrNonlinear(scip) ); /* nonlinear must be before linear, quadratic, abspower, and and due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrQuadratic(scip) ); /* quadratic must be before linear due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrLinear(scip) ); /* linear must be before its specializations due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrAbspower(scip) ); /* absolute power needs to be after quadratic and nonlinear due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrAnd(scip) );
   SCIP_CALL( SCIPincludeConshdlrBenders(scip) );
   SCIP_CALL( SCIPincludeConshdlrBenderslp(scip) );
   SCIP_CALL( SCIPincludeConshdlrBivariate(scip) ); /* bivariate needs to be after quadratic and nonlinear due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrBounddisjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrCardinality(scip) );
   SCIP_CALL( SCIPincludeConshdlrConjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrCountsols(scip) );
   SCIP_CALL( SCIPincludeConshdlrCumulative(scip) );
   SCIP_CALL( SCIPincludeConshdlrDisjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrIndicator(scip) );
   SCIP_CALL( SCIPincludeConshdlrIntegral(scip) );
   SCIP_CALL( SCIPincludeConshdlrKnapsack(scip) );
   SCIP_CALL( SCIPincludeConshdlrLinking(scip) );
   SCIP_CALL( SCIPincludeConshdlrLogicor(scip) );
   SCIP_CALL( SCIPincludeConshdlrOr(scip) );
   SCIP_CALL( SCIPincludeConshdlrOrbisack(scip) );
   SCIP_CALL( SCIPincludeConshdlrOrbitope(scip) );
   SCIP_CALL( SCIPincludeConshdlrPseudoboolean(scip) );
   SCIP_CALL( SCIPincludeConshdlrSetppc(scip) );
   SCIP_CALL( SCIPincludeConshdlrSOC(scip) ); /* SOC needs to be after quadratic due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrSOS1(scip) );
   SCIP_CALL( SCIPincludeConshdlrSOS2(scip) );
   SCIP_CALL( SCIPincludeConshdlrSuperindicator(scip) );
   SCIP_CALL( SCIPincludeConshdlrSymresack(scip) );
   SCIP_CALL( SCIPincludeConshdlrVarbound(scip) );
   SCIP_CALL( SCIPincludeConshdlrXor(scip) );
   SCIP_CALL( SCIPincludeConshdlrComponents(scip) );
   SCIP_CALL( SCIPincludeReaderBnd(scip) );
   SCIP_CALL( SCIPincludeReaderCcg(scip) );
   SCIP_CALL( SCIPincludeReaderCip(scip) );
   SCIP_CALL( SCIPincludeReaderCnf(scip) );
   SCIP_CALL( SCIPincludeReaderCor(scip) );
   SCIP_CALL( SCIPincludeReaderDiff(scip) );
   SCIP_CALL( SCIPincludeReaderFix(scip) );
   SCIP_CALL( SCIPincludeReaderFzn(scip) );
   SCIP_CALL( SCIPincludeReaderGms(scip) );
   SCIP_CALL( SCIPincludeReaderLp(scip) );
   SCIP_CALL( SCIPincludeReaderMps(scip) );
   SCIP_CALL( SCIPincludeReaderMst(scip) );
   SCIP_CALL( SCIPincludeReaderOpb(scip) );
   SCIP_CALL( SCIPincludeReaderOsil(scip) );
   SCIP_CALL( SCIPincludeReaderPip(scip) );
   SCIP_CALL( SCIPincludeReaderPpm(scip) );
   SCIP_CALL( SCIPincludeReaderPbm(scip) );
   SCIP_CALL( SCIPincludeReaderRlp(scip) );
   SCIP_CALL( SCIPincludeReaderSmps(scip) );
   SCIP_CALL( SCIPincludeReaderSol(scip) );
   SCIP_CALL( SCIPincludeReaderSto(scip) );
   SCIP_CALL( SCIPincludeReaderTim(scip) );
   SCIP_CALL( SCIPincludeReaderWbo(scip) );
   SCIP_CALL( SCIPincludeReaderZpl(scip) );
   SCIP_CALL( SCIPincludePresolBoundshift(scip) );
   SCIP_CALL( SCIPincludePresolConvertinttobin(scip) );
   SCIP_CALL( SCIPincludePresolDomcol(scip) );
   SCIP_CALL( SCIPincludePresolDualagg(scip) );
   SCIP_CALL( SCIPincludePresolDualcomp(scip) );
   SCIP_CALL( SCIPincludePresolDualinfer(scip) );
   SCIP_CALL( SCIPincludePresolGateextraction(scip) );
   SCIP_CALL( SCIPincludePresolImplics(scip) );
   SCIP_CALL( SCIPincludePresolInttobinary(scip) );
   SCIP_CALL( SCIPincludePresolQPKKTref(scip) );
   SCIP_CALL( SCIPincludePresolRedvub(scip) );
   SCIP_CALL( SCIPincludePresolTrivial(scip) );
   SCIP_CALL( SCIPincludePresolTworowbnd(scip) );
   SCIP_CALL( SCIPincludePresolSparsify(scip) );
   SCIP_CALL( SCIPincludePresolStuffing(scip) );
   SCIP_CALL( SCIPincludePresolSymmetry(scip) );
   SCIP_CALL( SCIPincludePresolSymbreak(scip) );   /* needs to be included after presol_symmetry */
   SCIP_CALL( SCIPincludeNodeselBfs(scip) );
   SCIP_CALL( SCIPincludeNodeselBreadthfirst(scip) );
   SCIP_CALL( SCIPincludeNodeselDfs(scip) );
   SCIP_CALL( SCIPincludeNodeselEstimate(scip) );
   SCIP_CALL( SCIPincludeNodeselHybridestim(scip) );
   SCIP_CALL( SCIPincludeNodeselRestartdfs(scip) );
   SCIP_CALL( SCIPincludeNodeselUct(scip) );
   SCIP_CALL( SCIPincludeBranchruleAllfullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleCloud(scip) );
   SCIP_CALL( SCIPincludeBranchruleDistribution(scip) );
   SCIP_CALL( SCIPincludeBranchruleFullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleInference(scip) );
   SCIP_CALL( SCIPincludeBranchruleLeastinf(scip) );
   SCIP_CALL( SCIPincludeBranchruleLookahead(scip) );
   SCIP_CALL( SCIPincludeBranchruleMostinf(scip) );
   SCIP_CALL( SCIPincludeBranchruleMultAggr(scip) );
   SCIP_CALL( SCIPincludeBranchruleNodereopt(scip) );
   SCIP_CALL( SCIPincludeBranchrulePscost(scip) );
   SCIP_CALL( SCIPincludeBranchruleRandom(scip) );
   SCIP_CALL( SCIPincludeBranchruleRelpscost(scip) );
   SCIP_CALL( SCIPincludeEventHdlrSolvingphase(scip) );
   SCIP_CALL( SCIPincludeComprLargestrepr(scip) );
   SCIP_CALL( SCIPincludeComprWeakcompr(scip) );
   SCIP_CALL( SCIPincludePropDualfix(scip) );
   SCIP_CALL( SCIPincludePropGenvbounds(scip) );
   SCIP_CALL( SCIPincludePropObbt(scip) );
   SCIP_CALL( SCIPincludePropOrbitalfixing(scip) );
   SCIP_CALL( SCIPincludePropNlobbt(scip) );
   SCIP_CALL( SCIPincludePropProbing(scip) );
   SCIP_CALL( SCIPincludePropPseudoobj(scip) );
   SCIP_CALL( SCIPincludePropRedcost(scip) );
   SCIP_CALL( SCIPincludePropRootredcost(scip) );
   SCIP_CALL( SCIPincludePropVbounds(scip) );
   SCIP_CALL( SCIPincludeSepaCGMIP(scip) );
   SCIP_CALL( SCIPincludeSepaClique(scip) );
   SCIP_CALL( SCIPincludeSepaClosecuts(scip) );
   SCIP_CALL( SCIPincludeSepaAggregation(scip) );
   SCIP_CALL( SCIPincludeSepaConvexproj(scip) );
   SCIP_CALL( SCIPincludeSepaDisjunctive(scip) );
   SCIP_CALL( SCIPincludeSepaEccuts(scip) );
   SCIP_CALL( SCIPincludeSepaGauge(scip) );
   SCIP_CALL( SCIPincludeSepaGomory(scip) );
   SCIP_CALL( SCIPincludeSepaImpliedbounds(scip) );
   SCIP_CALL( SCIPincludeSepaIntobj(scip) );
   SCIP_CALL( SCIPincludeSepaMcf(scip) );
   SCIP_CALL( SCIPincludeSepaOddcycle(scip) );
   SCIP_CALL( SCIPincludeSepaRapidlearning(scip) );
   SCIP_CALL( SCIPincludeSepaStrongcg(scip) );
   SCIP_CALL( SCIPincludeSepaZerohalf(scip) );
   SCIP_CALL( SCIPincludeDispDefault(scip) );
   SCIP_CALL( SCIPincludeTableDefault(scip) );
   // SCIP_CALL( SCIPincludeEventHdlrSofttimelimit(scip) );
   SCIP_CALL( SCIPincludeConcurrentScipSolvers(scip) );
   SCIP_CALL( SCIPincludeBendersDefault(scip) );
}