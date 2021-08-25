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

/**@file   heur_MLdfs.c
 * @brief  Local branching heuristic according to Fischetti and Lodi
 * @author Timo Berthold
 * @author Marc Pfetsch
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <blockmemshell/memory.h>
#include <scip/cons_linear.h>
#include <scip/heuristics.h>
#include "heur_ml_subscip.hpp"
#include <scip/pub_event.h>
#include <scip/pub_heur.h>
#include <scip/pub_message.h>
#include <scip/pub_misc.h>
#include <scip/pub_sol.h>
#include <scip/pub_var.h>
#include <scip/scip_branch.h>
#include <scip/scip_cons.h>
#include <scip/scip_copy.h>
#include <scip/scip_event.h>
#include <scip/scip_general.h>
#include <scip/scip_heur.h>
#include <scip/scip_mem.h>
#include <scip/scip_message.h>
#include <scip/scip_nodesel.h>
#include <scip/scip_numerics.h>
#include <scip/scip_param.h>
#include <scip/scip_prob.h>
#include <scip/scip_sol.h>
#include <scip/scip_solve.h>
#include <scip/scip_solvingstats.h>
#include <string.h>
#include "branching_ml_dfs.hpp"
#include "nodesel_ml_dfs.hpp"

#define HEUR_NAME             "mldfs"
#define HEUR_DESC             "local branching heuristic by Fischetti and Lodi"
#define HEUR_DISPCHAR         'L'
#define HEUR_PRIORITY         -1000000
#define HEUR_FREQ             100
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_BEFOREPRESOL
#define HEUR_USESSUBSCIP      TRUE  /**< does the heuristic use a secondary SCIP instance? */

#define DEFAULT_NEIGHBORHOODSIZE  18    /* radius of the incumbents neighborhood to be searched                     */
#define DEFAULT_NODESOFS      1000      /* number of nodes added to the contingent of the total nodes               */
#define DEFAULT_MAXNODES      10000     /* maximum number of nodes to regard in the subproblem                      */
#define DEFAULT_MINIMPROVE    0.01      /* factor by which MLdfs should at least improve the incumbent     */
#define DEFAULT_MINNODES      1000      /* minimum number of nodes required to start the subproblem                 */
#define DEFAULT_NODESQUOT     0.05      /* contingent of sub problem nodes in relation to original nodes            */
#define DEFAULT_LPLIMFAC      1.5       /* factor by which the limit on the number of LP depends on the node limit  */
#define DEFAULT_NWAITINGNODES 200       /* number of nodes without incumbent change that heuristic should wait      */
#define DEFAULT_USELPROWS     FALSE     /* should subproblem be created out of the rows in the LP rows,
                                         * otherwise, the copy constructors of the constraints handlers are used    */
#define DEFAULT_COPYCUTS      TRUE      /* if DEFAULT_USELPROWS is FALSE, then should all active cuts from the cutpool
                                         * of the original scip be copied to constraints of the subscip
                                         */
#define DEFAULT_BESTSOLLIMIT   3         /* limit on number of improving incumbent solutions in sub-CIP            */
#define DEFAULT_USEUCT        FALSE     /* should uct node selection be used at the beginning of the search?     */

/* event handler properties */
#define EVENTHDLR_NAME         "MLdfs"
#define EVENTHDLR_DESC         "LP event handler for " HEUR_NAME " heuristic"


#define EXECUTE               0
#define WAITFORNEWSOL         1


/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   int                   nwaitingnodes;      /**< number of nodes without incumbent change that heuristic should wait  */
   int                   nodesofs;           /**< number of nodes added to the contingent of the total nodes           */
   int                   minnodes;           /**< minimum number of nodes required to start the subproblem             */
   int                   maxnodes;           /**< maximum number of nodes to regard in the subproblem                  */
   SCIP_Longint          usednodes;          /**< amount of nodes local branching used during all calls                */
   SCIP_Real             nodesquot;          /**< contingent of sub problem nodes in relation to original nodes        */
   SCIP_Real             minimprove;         /**< factor by which MLdfs should at least improve the incumbent */
   SCIP_Real             nodelimit;          /**< the nodelimit employed in the current sub-SCIP, for the event handler*/
   SCIP_Real             lplimfac;           /**< factor by which the limit on the number of LP depends on the node limit */
   int                   neighborhoodsize;   /**< radius of the incumbent's neighborhood to be searched                */
   int                   callstatus;         /**< current status of MLdfs heuristic                           */
   SCIP_SOL*             lastsol;            /**< the last incumbent MLdfs used as reference point            */
   int                   curneighborhoodsize;/**< current neighborhoodsize                                             */
   int                   curminnodes;        /**< current minimal number of nodes required to start the subproblem     */
   int                   emptyneighborhoodsize;/**< size of neighborhood that was proven to be empty                   */
   SCIP_Bool             uselprows;          /**< should subproblem be created out of the rows in the LP rows?         */
   SCIP_Bool             copycuts;           /**< if uselprows == FALSE, should all active cuts from cutpool be copied
                                              *   to constraints in subproblem?
                                              */
   int                   bestsollimit;       /**< limit on number of improving incumbent solutions in sub-CIP            */
   SCIP_Bool             useuct;             /**< should uct node selection be used at the beginning of the search?  */
};

/** creates a new solution for the original problem by copying the solution of the subproblem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 scip,               /**< SCIP data structure  of the original problem      */
   SCIP*                 subscip,            /**< SCIP data structure  of the subproblem            */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                     */
   SCIP_HEUR*            heur,               /**< the MLdfs heuristic                      */
   SCIP_SOL*             subsol,             /**< solution of the subproblem                          */
   SCIP_Bool*            success             /**< pointer to store, whether new solution was found  */
   )
{
   SCIP_VAR** vars;
   int nvars;
   SCIP_SOL* newsol;
   SCIP_Real* subsolvals;

   assert( scip != NULL );
   assert( subscip != NULL );
   assert( subvars != NULL );
   assert( subsol != NULL );

   /* copy the solution */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   /* sub-SCIP may have more variables than the number of active (transformed) variables in the main SCIP
    * since constraint copying may have required the copy of variables that are fixed in the main SCIP
    */
   assert(nvars <= SCIPgetNOrigVars(subscip));

   SCIP_CALL( SCIPallocBufferArray(scip, &subsolvals, nvars) );

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(subscip, subsol, nvars, subvars, subsolvals) );

   /* create new solution for the original problem */
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newsol, nvars, vars, subsolvals) );

   SCIP_CALL( SCIPtrySolFree(scip, &newsol, FALSE, FALSE, TRUE, TRUE, TRUE, success) );

   SCIPfreeBufferArray(scip, &subsolvals);

   return SCIP_OKAY;
}


/* ---------------- Callback methods of event handler ---------------- */

/* exec the event handler
 *
 * we interrupt the solution process
 */
static
SCIP_DECL_EVENTEXEC(eventExecMLdfs)
{
   SCIP_HEURDATA* heurdata;

   assert(eventhdlr != NULL);
   assert(eventdata != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);
   assert(event != NULL);
   assert(SCIPeventGetType(event) & SCIP_EVENTTYPE_LPSOLVED);

   heurdata = (SCIP_HEURDATA*)eventdata;
   assert(heurdata != NULL);

   /* interrupt solution process of sub-SCIP */
   if( SCIPgetNLPs(scip) > heurdata->lplimfac * heurdata->nodelimit )
   {
      SCIPdebugMsg(scip, "interrupt after  %" SCIP_LONGINT_FORMAT " LPs\n",SCIPgetNLPs(scip));
      SCIP_CALL( SCIPinterruptSolve(scip) );
   }

   return SCIP_OKAY;
}


/*
 * Callback methods of primal heuristic
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
static
SCIP_DECL_HEURCOPY(heurCopyMLdfs)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* call inclusion method of primal heuristic */
   SCIP_CALL( SCIPincludeHeurMLdfs(scip) );

   return SCIP_OKAY;
}

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeMLdfs)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert( heur != NULL );
   assert( scip != NULL );

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );

   /* free heuristic data */
   SCIPfreeBlockMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitMLdfs)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert( heur != NULL );
   assert( scip != NULL );

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );

   /* with a little abuse we initialize the heurdata as if MLdfs would have finished its last step regularly */
   heurdata->callstatus = WAITFORNEWSOL;
   heurdata->lastsol = NULL;
   heurdata->usednodes = 0;
   heurdata->curneighborhoodsize = heurdata->neighborhoodsize;
   heurdata->curminnodes = heurdata->minnodes;
   heurdata->emptyneighborhoodsize = 0;

   return SCIP_OKAY;
}


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecMLdfs)
{  /*lint --e{715}*/
   SCIP_VAR** subvars;                       /* subproblem's variables                                */
   SCIP_EVENTHDLR* eventhdlr;                /* event handler for LP events                     */
   SCIP_HEURDATA* heurdata;
   SCIP_HASHMAP* varmapfw;                   /* mapping of SCIP variables to sub-SCIP variables */
   SCIP_VAR** vars;

   SCIP_Real cutoff;                         /* objective cutoff for the subproblem                   */
   SCIP_Real upperbound;
   int nvars;
   int i;
   SCIP_Bool success;
   SCIP* subscip;
   SCIP_CALL( SCIPcreate(&subscip) );

   printf("exe ml dfs heuristic\n");
   /* get the data of the variables and the best solution */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );

   /* create the variable mapping hash map */
   SCIP_CALL( SCIPhashmapCreate(&varmapfw, SCIPblkmem(subscip), nvars) );
   success = FALSE;

   SCIPcopy(scip, subscip, varmapfw, NULL, "df", FALSE, FALSE, FALSE, 0, NULL);

   SCIP_CALL( SCIPallocBufferArray(scip, &subvars, nvars) );
   for (i = 0; i < nvars; ++i)
      subvars[i] = (SCIP_VAR*) SCIPhashmapGetImage(varmapfw, vars[i]);

   assert(scip != NULL);
   assert(subscip != NULL);

   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "timing/statistictiming", FALSE) );
   /* speed up the heuristic */
   SCIPsetIntParam(subscip, "presolving/maxrounds", 0);
   SCIPsetIntParam(subscip, "separating/maxrounds", 0);
   SCIPsetIntParam(subscip, "presolving/maxrestarts", 0);
   SCIPsetIntParam(subscip, "separating/maxroundsroot", 0);
   SCIPsetHeuristics(subscip, SCIP_PARAMSETTING_OFF, TRUE);
   SCIP_CALL( SCIPsetSubscipsOff(subscip, TRUE) );

   /* set termination condition */
   // SCIPsetRealParam(subscip, "limits/time", 5);
   SCIPsetIntParam(subscip, "limits/solutions", 1);

   COML::Branching_dfs* branching_policy = new COML::Branching_dfs( subscip, "", "", 66666667, -1, 1);
   COML::SCIPincludeObjBranchrule_MLDFS(subscip, branching_policy, false);

   COML::Nodesel_ML_DFS* nodesel_policy = new COML::Nodesel_ML_DFS(subscip, "", "", 66666667, 66666667);
   SCIPincludeObjNodesel_MLDFS(subscip, nodesel_policy, false);
   SCIP_CALL( SCIPtransformProb(subscip) );
   SCIP_CALL_ABORT( SCIPsolve(subscip) );
   printf("endsolve\n");
   if( SCIPgetNSols(subscip) > 0 )
   {
      SCIP_SOL** subsols;
      int nsubsols;

      /* check, whether a solution was found;
      * due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted
      */
      nsubsols = SCIPgetNSols(subscip);
      subsols = SCIPgetSols(subscip);
      success = FALSE;
      for( i = 0; i < nsubsols; ++i )
      {
         SCIP_CALL( createNewSol(scip, subscip, subvars, heur, subsols[i], &success) );
         if( success )
         {   
               *result = SCIP_FOUNDSOL;
         }
      }
   }
   else
   {
      printf("heuristic did not find any feasible solution\n");
   }

   /* free subproblem */
   SCIPfreeBufferArray(scip, &subvars);
   SCIP_CALL( SCIPfree(&subscip) );
   return SCIP_OKAY;
}


/*
 * primal heuristic specific interface methods
 */

/** creates the MLdfs primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurMLdfs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_HEUR* heur;

   /* create MLdfs primal heuristic data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecMLdfs, heurdata) );

   assert(heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurCopy(scip, heur, heurCopyMLdfs) );
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeMLdfs) );
   SCIP_CALL( SCIPsetHeurInit(scip, heur, heurInitMLdfs) );

   return SCIP_OKAY;
}
