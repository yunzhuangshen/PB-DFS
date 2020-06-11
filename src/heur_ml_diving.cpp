/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2019 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   heur_guideddiving.c
 * @brief  LP diving heuristic that chooses fixings in direction of incumbent solutions
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "heur_ml_diving.hpp"
#include "global_config.hpp"
#include <scip/heuristics.h>
#include <scip/pub_heur.h>
#include <scip/pub_message.h>
#include <scip/pub_sol.h>
#include <scip/pub_var.h>
#include <scip/scip_heur.h>
#include <scip/scip_lp.h>
#include <scip/scip_mem.h>
#include <scip/scip_numerics.h>
#include <scip/scip_prob.h>
#include <scip/scip_sol.h>
#include <string.h>

#define HEUR_NAME             "mldiving"
#define HEUR_DESC             "LP diving heuristic that chooses fixings in direction of incumbent solutions"
#define HEUR_DISPCHAR         'g'
#define HEUR_PRIORITY         -1000000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERLPPLUNGE
#define HEUR_USESSUBSCIP      FALSE  /**< does the heuristic use a secondary SCIP instance? */
#define DIVESET_DIVETYPES     SCIP_DIVETYPE_INTEGRALITY /**< bit mask that represents all supported dive types */


/*
 * Default parameter settings
 */

#define DEFAULT_MINRELDEPTH         0.0 /**< minimal relative depth to start diving */
#define DEFAULT_MAXRELDEPTH         1.0 /**< maximal relative depth to start diving */
#define DEFAULT_MAXLPITERQUOT      0.05 /**< maximal fraction of diving LP iterations compared to node LP iterations */
#define DEFAULT_MAXLPITEROFS       1000 /**< additional number of allowed LP iterations */
#define DEFAULT_MAXDIVEUBQUOT       0.8 /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOT      0.0 /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_BACKTRACK          TRUE /**< use one level of backtracking if infeasibility is encountered? */
#define DEFAULT_LPRESOLVEDOMCHGQUOT 0.15 /**< percentage of immediate domain changes during probing to trigger LP resolve */
#define DEFAULT_LPSOLVEFREQ           0 /**< LP solve frequency for diving heuristics */
#define DEFAULT_ONLYLPBRANCHCANDS FALSE /**< should only LP branching candidates be considered instead of the slower but
                                         *   more general constraint handler diving variable selection? */
#define DEFAULT_RANDSEED            127 /**< initial seed for random number generation */

/* locally defined heuristic data */
struct SCIP_HeurData
{
   SCIP_SOL*             sol;                /**< working solution */
};

/*
 * local methods
 */

/*
 * Callback methods
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
static
SCIP_DECL_HEURCOPY(heurCopyML)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* call inclusion method of primal heuristic */
   SCIP_CALL( SCIPincludeHeurMLdiving(scip) );

   return SCIP_OKAY;
}

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeML) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);

   /* free heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   SCIPfreeBlockMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitML) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* create working solution */
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitML) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free working solution */
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );

   return SCIP_OKAY;
}


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecML) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;
   SCIP_DIVESET* diveset;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);
   assert(SCIPhasCurrentNodeLP(scip));

   *result = SCIP_DIDNOTRUN;

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* if there are no integer variables (note that, e.g., SOS1 variables may be present) */
   if ( SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip) == 0 )
      return SCIP_OKAY;

   assert(SCIPheurGetNDivesets(heur) > 0);
   assert(SCIPheurGetDivesets(heur) != NULL);
   diveset = SCIPheurGetDivesets(heur)[0];
   assert(diveset != NULL);

   /* call generic diving algorithm */
   SCIP_CALL( SCIPperformGenericDivingAlgorithm(scip, diveset, heurdata->sol, heur, result, nodeinfeasible) );

   return SCIP_OKAY;
}

/* callbacks for diving */

/** calculate score and preferred rounding direction for the candidate variable; the best candidate maximizes the
 *  score
 */
static
SCIP_DECL_DIVESETGETSCORE(divesetGetScoreML)
{
   SCIP_SOL* bestsol;
   SCIP_Real bestsolval;
   SCIP_Real obj;
   SCIP_Real objnorm;
   SCIP_Real objgain;

   
   const char* var_full_name = SCIPvarGetName(cand);
   std::string var_name(var_full_name + gconf.prefix_size_remove, strlen(var_full_name) - gconf.prefix_size_remove);
   double prob_1 = gconf.ml_scores1[gconf.var_mapper[var_name]];

   /* calculate objective gain and fractionality for the selected rounding direction */
   // if( gconf.prob == 0 || gconf.prob == 1)
   // {
   //    *score = MAX(1 - prob_1, prob_1);
   //    *roundup = prob_1 >= 0.5;

   // }
   // else
   // {
      *score = prob_1;
      *roundup = prob_1 >= 0.5;
   // }

   /* prefer decisions on binary variables */
   if( !SCIPvarIsBinary(cand) )
      *score *= 0.1;

   return SCIP_OKAY;
}

/*
 * heuristic specific interface methods
 */

/** creates the guideddiving heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurMLdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_HEUR* heur;

   /* create ML primal heuristic data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecML, heurdata) );
   assert(heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurCopy(scip, heur, heurCopyML) );
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeML) );
   SCIP_CALL( SCIPsetHeurInit(scip, heur, heurInitML) );
   SCIP_CALL( SCIPsetHeurExit(scip, heur, heurExitML) );

   /* create a diveset (this will automatically install some additional parameters for the heuristic)*/
   SCIP_CALL( SCIPcreateDiveset(scip, NULL, heur, HEUR_NAME, DEFAULT_MINRELDEPTH, DEFAULT_MAXRELDEPTH, DEFAULT_MAXLPITERQUOT,
         DEFAULT_MAXDIVEUBQUOT, DEFAULT_MAXDIVEAVGQUOT, 1.0, 1.0, DEFAULT_LPRESOLVEDOMCHGQUOT, DEFAULT_LPSOLVEFREQ,
         DEFAULT_MAXLPITEROFS, DEFAULT_RANDSEED, DEFAULT_BACKTRACK, DEFAULT_ONLYLPBRANCHCANDS, DIVESET_DIVETYPES, divesetGetScoreML) );

   return SCIP_OKAY;
}
