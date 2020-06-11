

#include "branching_policy.hpp"
#include "utils.hpp"



#define DEFAULT_REEVALAGE 10LL
#define DEFAULT_MAXPROPROUNDS -2
#define DEFAULT_PROBINGBOUNDS TRUE

#define SIMPLEX_ITERATION_LIMIT 500
#define DEFAULT_MAXLOOKAHEAD 9 /**< maximal number of further variables evaluated without better score */
#define DEFAULT_INITCAND 30    /**< maximal number of candidates initialized with strong branching per node */

/** branching rule data */
struct SCIP_BranchruleData
{
   scip::ObjBranchrule *objbranchrule; /**< branching rule object */
   SCIP_Bool deleteobject;             /**< should the branching rule object be deleted when branching rule is freed? */
};

static SCIP_RETCODE printNodeRootPath(
    SCIP *scip,      /**< SCIP data structure */
    SCIP_NODE *node, /**< node data */
    FILE *file       /**< output file (or NULL for standard output) */
);


/*
 *                                 actual methods that matters
 * --------------------------------------------------------------------------------------------------------------
 */

/**
 * called back from SCIP every branching decision
 * 
 * arguments: 
 *    SCIP* scip;
 *    SCIP_BRANCHRULE *branchrule;
 *    unsigned int allowaddcons;
 *    SCIP_RESULT *result;
 * */

/**
 * if current node depth less than the strong branching depth,
 * we perform strong branching and update variable statistics.
 * otherwise we train the model and let the model select branching variables.
 * 
*/

COML::Branching::Branching(
    // ----------------------------------------------  SCIP parameters  --------------------------------------------
    SCIP *scip,            /**< SCIP data structure */
    const char *name,      /**< name of branching rule */
    const char *desc,      /**< description of branching rule */
    int priority,          /**< priority of the branching rule */
    int maxdepth,          /**< maximal depth level, up to which this branching rule should be used (or -1) */
    double maxbounddist /**< maximal relative distance from current node's dual bound to primal bound
                                              *   compared to best node's dual bound for applying branching rule
                                              *   (0.0: only on current best node, 1.0: on all nodes) */

    // ----------------------------------------------  ml parameters  --------------------------------------------
    ) : ObjBranchrule(scip, name, desc, priority, maxdepth, maxbounddist){};
         
SCIP_DECL_BRANCHINIT(COML::Branching::scip_init)
{
   int num_vars;
   int num_binary_vars;
   int num_integral_vars;
   int num_implicit_integral_vars;
   int num_coutinous_vars;
   SCIPgetVarsData(scip, NULL, &num_vars, &num_binary_vars,
                   &num_integral_vars, &num_implicit_integral_vars, &num_coutinous_vars);
   printf("statistics of the transformed problem: \
          num_vars: %d, num_binary_vars: %d, num_integral_vars: %d, \
          num_implicit_integral_vars: %d, num_coutinous_vars: %d \n",
          num_vars, num_binary_vars, num_integral_vars, num_implicit_integral_vars, num_coutinous_vars);

   return SCIP_OKAY;
}

void COML::Branching::record_solving_curves(SCIP* scip)
{
   // record primal info for curve plot
   SCIP_SOL* sol = SCIPgetBestSol(scip);
   double time = SCIPgetSolvingTime(scip);
   if(sol != NULL)
   {
      SCIP_HEUR* h = SCIPsolGetHeur(sol);
      double obj = SCIPgetSolOrigObj(scip, sol);
      
      if (gconf.objs.empty())
      {  

         gconf.objs.push_back(obj);
         gconf.time.push_back(SCIPsolGetTime(sol));
         gconf.first_sol_heur = h == NULL ? "none":SCIPheurGetName(h);
         printf("first solution: %f %fs\n", obj, time);
         if (h!=NULL)
            printf("heuristic: %s\n", SCIPheurGetName(h));
         else
            printf("heuristic is None\n");
      }else if (
            (gconf.optimization_type == Optimization_Type::Maximization && 
                  obj > gconf.objs[gconf.objs.size()-1] + SCIPsetEpsilon(scip->set)) ||
            (gconf.optimization_type == Optimization_Type::Minimization && 
                  obj < gconf.objs[gconf.objs.size()-1] - SCIPsetEpsilon(scip->set))
      ){

         

         gconf.objs.push_back(obj);
         gconf.time.push_back(time);
         printf("better solution: %f %fs\n", obj, time);
         if (h!=NULL)
            printf("heuristic: %s\n", SCIPheurGetName(h));
         else
            printf("heuristic is None\n");
      }
   }

   // record dual info for curve plot
   double dualbound = SCIPgetDualbound(scip);
   if (gconf.dual_bounds.empty())
   {
      gconf.dual_bounds.push_back(dualbound);
      gconf.dual_time.push_back(time);
   }else if (
      (gconf.optimization_type == Optimization_Type::Maximization && 
            dualbound < gconf.dual_bounds[gconf.dual_bounds.size()-1] - SCIPsetEpsilon(scip->set)) ||
      (gconf.optimization_type == Optimization_Type::Minimization && 
            dualbound > gconf.dual_bounds[gconf.dual_bounds.size()-1] + SCIPsetEpsilon(scip->set))
   ){
      gconf.dual_bounds.push_back(dualbound);
      gconf.dual_time.push_back(time);
   }

   // record gap info for curve plot
   double optgap = SCIPgetGap(scip);
   if (gconf.gaps.empty()){
      gconf.gaps.push_back(optgap);
      gconf.gap_time.push_back(time);
   }else if (optgap < gconf.gaps[gconf.gaps.size()-1] - SCIPsetEpsilon(scip->set)){
      gconf.gaps.push_back(optgap);
      gconf.gap_time.push_back(time);
   }

}

SCIP_DECL_BRANCHEXECLP(COML::Branching::scip_execlp)
{

   if (gconf.record_solving_info)
      record_solving_curves(scip);

   SCIP_BRANCHRULEDATA *branchruledata;
   SCIP_VAR **tmplpcands;
   SCIP_VAR **lpcands;
   double *tmplpcandsfrac;
   double *lpcandsfrac;
   double bestdown;
   double bestup;
   double provedbound;
   SCIP_Bool bestdownvalid;
   SCIP_Bool bestupvalid;
   int npriolpcands;
   double *tmplpcandssol;
   double *lpcandssol;
   double bestscore;
   int nlpcands;
   int bestcand = -1;


   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), this->scip_name_) == 0);
   assert(scip != NULL);
   assert(scip == this->scip_);
   assert(result != NULL);
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   *result = SCIP_DIDNOTRUN;
   SCIP_CALL(SCIPgetLPBranchCands(scip, &tmplpcands, &tmplpcandssol, &tmplpcandsfrac, &nlpcands, &npriolpcands, NULL));
   assert(nlpcands > 0);
   assert(npriolpcands > 0);
   SCIP_CALL(SCIPduplicateBufferArray(scip, &lpcands, tmplpcands, nlpcands));
   SCIP_CALL(SCIPduplicateBufferArray(scip, &lpcandssol, tmplpcandssol, nlpcands));
   SCIP_CALL(SCIPduplicateBufferArray(scip, &lpcandsfrac, tmplpcandsfrac, nlpcands));

   SCIP_NODE* focus_node = SCIPgetFocusNode(scip);
   long long cur_node_id = SCIPnodeGetNumber(focus_node);
   long long nnodes = SCIPgetNNodes(scip) - 1;
   int nnodeLeft = SCIPgetNNodesLeft(scip);
   // printf("cur_node_id: %lld\n", cur_node_id);
   // printf("global_cut off bound: %f\n\n", SCIPgetCutoffbound(scip_));
   // printf("is dual node: %d\n", gconf.is_dual_node(cur_node_id));

   if (gconf.policy == Policy::ML_DING)
   {
      if (cur_node_id == 1)
      {
         SCIP_NODE* left_child_ptr; SCIP_NODE* right_child_ptr; 
         gconf.create_children_by_ml_local_cuts(scip, &left_child_ptr, &right_child_ptr, false);
         (*result) = SCIP_BRANCHED;
      }
      else
      {
         SCIPexecRelpscostBranching(scip, lpcands, lpcandssol, lpcandsfrac, nlpcands,
                              TRUE, result);
      }
   }else if (gconf.policy == Policy::ML_DFS_EXACT_SCORE1_GCN ||
               gconf.policy == Policy::ML_DFS_EXACT_SCORE2_GCN || 
               gconf.policy == Policy::ML_DFS_EXACT_SCORE3_GCN){

      double cands_ml_score[nlpcands];
      const char* var_name;
      int var_idx;
      char var_short_name[10];
      int prefix_remove =  gconf.prefix_size_remove;
      if (gconf.policy==Policy::ML_DFS_EXACT_SCORE1_GCN)
      {
         for (int i = 0; i < nlpcands; i++)
         {
            var_name = SCIPvarGetName(lpcands[i]);
            strcpy(var_short_name, var_name + prefix_remove);
            if (gconf.var_mapper.find(var_short_name) == gconf.var_mapper.end())
            {
               cands_ml_score[i] = 0;
            }
            else
            {
               var_idx = gconf.var_mapper[var_short_name];
               cands_ml_score[i] = gconf.ml_scores1[var_idx];
            }
         }
      } 
      else if (gconf.policy==Policy::ML_DFS_EXACT_SCORE3_GCN)
      {
         for (int i = 0; i < nlpcands; i++)
         {
            var_name = SCIPvarGetName(lpcands[i]);
            strcpy(var_short_name, var_name + prefix_remove);

            if (gconf.var_mapper.find(var_short_name) == gconf.var_mapper.end())
            {
               cands_ml_score[i] = 0;
            }
            else
            {
               var_idx = gconf.var_mapper[var_short_name];
               cands_ml_score[i] = 1-gconf.ml_scores1[var_idx];
            }
         }
      }
      else if (gconf.policy==Policy::ML_DFS_EXACT_SCORE2_GCN )
      {
         for (int i = 0; i < nlpcands; i++)
         {
            var_name = SCIPvarGetName(lpcands[i]);
            strcpy(var_short_name, var_name + prefix_remove);

            if (gconf.var_mapper.find(var_short_name) == gconf.var_mapper.end())
            {
               cands_ml_score[i] = 0;
            }
            else
            {
               var_idx = gconf.var_mapper[var_short_name];
               cands_ml_score[i] = gconf.ml_scores2[var_idx];
            }
         }
      }
      bestcand = COML::calc_argmax(cands_ml_score, nlpcands);
      // printf("bestcand: %d, bestcand score: %f\n", bestcand, cands_ml_score[bestcand]);
      assert(*result == SCIP_DIDNOTRUN);
      if (0 > bestcand || bestcand >= nlpcands)
         printf("bestcand: %d, nlpcands: %d\n", bestcand, nlpcands);
      assert(0 <= bestcand && bestcand < nlpcands);

      SCIP_NODE *downchild; SCIP_NODE *upchild;
      SCIP_CALL(SCIPbranchVar(this->scip_, lpcands[bestcand], &downchild, NULL, &upchild));
      
      if (downchild == NULL && upchild == NULL)
         *result = SCIP_CUTOFF;
      else
         *result = SCIP_BRANCHED;
      }
   else {
      SCIPexecRelpscostBranching(scip, lpcands, lpcandssol, lpcandsfrac, nlpcands,
                              TRUE, result);
   }
   

   // printf("result code: %d\n", *result);
   SCIPfreeBufferArray(scip, &lpcandsfrac);
   SCIPfreeBufferArray(scip, &lpcandssol);
   SCIPfreeBufferArray(scip, &lpcands);

   return SCIP_OKAY;
}


COML::Branching::~Branching()
{
}


/*
 *                                         Callback methods of SCIP
 * --------------------------------------------------------------------------------------------------------------
 */
extern "C"
{
   static SCIP_DECL_BRANCHCOPY(branchCopyDynamicBranching);
   static SCIP_DECL_BRANCHFREE(branchFreeDynamicBranching);
   static SCIP_DECL_BRANCHINIT(branchInitDynamicBranching);
   static SCIP_DECL_BRANCHEXIT(branchExitDynamicBranching);
   static SCIP_DECL_BRANCHINITSOL(branchInitsolDynamicBranching);
   static SCIP_DECL_BRANCHEXITSOL(branchExitsolDynamicBranching);
   static SCIP_DECL_BRANCHEXECLP(branchExeclpDynamicBranching);
   static SCIP_DECL_BRANCHEXECEXT(branchExecextDynamicBranching);
   static SCIP_DECL_BRANCHEXECPS(branchExecpsDynamicBranching);
}

/** creates the branching rule for the given branching rule object and includes it in SCIP */
SCIP_RETCODE COML::SCIPincludeObjBranchrule_DB(
    SCIP *scip,                         /**< SCIP data structure */
    scip::ObjBranchrule *objbranchrule, /**< branching rule object */
    SCIP_Bool deleteobject              /**< should the branching rule object be deleted when branching rule is freed? */
)
{
   SCIP_BRANCHRULEDATA *branchruledata;

   assert(scip != NULL);
   assert(objbranchrule != NULL);

   /* create branching rule data */
   branchruledata = new SCIP_BRANCHRULEDATA;
   branchruledata->objbranchrule = objbranchrule;
   branchruledata->deleteobject = deleteobject;

   /* include branching rule */
   SCIP_CALL(SCIPincludeBranchrule(scip, objbranchrule->scip_name_, objbranchrule->scip_desc_,
                                   objbranchrule->scip_priority_, objbranchrule->scip_maxdepth_, objbranchrule->scip_maxbounddist_,
                                   branchCopyDynamicBranching,
                                   branchFreeDynamicBranching, branchInitDynamicBranching, branchExitDynamicBranching, branchInitsolDynamicBranching, branchExitsolDynamicBranching,
                                   branchExeclpDynamicBranching, branchExecextDynamicBranching, branchExecpsDynamicBranching,
                                   branchruledata)); /*lint !e429*/

   return SCIP_OKAY; /*lint !e429*/
}

extern "C"
{

   /** copy method for branchrule plugins (called when SCIP copies plugins) */
   static SCIP_DECL_BRANCHCOPY(branchCopyDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      assert(scip != NULL);

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      assert(branchruledata->objbranchrule->scip_ != scip);

      if (branchruledata->objbranchrule->iscloneable())
      {
         scip::ObjBranchrule *newobjbranchrule;
         newobjbranchrule = dynamic_cast<scip::ObjBranchrule *>(branchruledata->objbranchrule->clone(scip));

         /* call include method of branchrule object */
         SCIP_CALL(COML::SCIPincludeObjBranchrule_DB(scip, newobjbranchrule, TRUE));
      }

      return SCIP_OKAY;
   }

   /** destructor of branching rule to free user data (called when SCIP is exiting) */
   static SCIP_DECL_BRANCHFREE(branchFreeDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      assert(branchruledata->objbranchrule->scip_ == scip);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_free(scip, branchrule));

      /* free branchrule object */
      if (branchruledata->deleteobject)
         delete branchruledata->objbranchrule;

      /* free branchrule data */
      delete branchruledata;
      SCIPbranchruleSetData(branchrule, NULL); /*lint !e64*/

      return SCIP_OKAY;
   }

   /** initialization method of branching rule (called after problem was transformed) */
   static SCIP_DECL_BRANCHINIT(branchInitDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      assert(branchruledata->objbranchrule->scip_ == scip);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_init(scip, branchrule));

      return SCIP_OKAY;
   }

   /** deinitialization method of branching rule (called before transformed problem is freed) */
   static SCIP_DECL_BRANCHEXIT(branchExitDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_exit(scip, branchrule));

      return SCIP_OKAY;
   }

   /** solving process initialization method of branching rule (called when branch and bound process is about to begin) */
   static SCIP_DECL_BRANCHINITSOL(branchInitsolDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_initsol(scip, branchrule));

      return SCIP_OKAY;
   }

   /** solving process deinitialization method of branching rule (called before branch and bound process data is freed) */
   static SCIP_DECL_BRANCHEXITSOL(branchExitsolDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_exitsol(scip, branchrule));

      return SCIP_OKAY;
   }

   /** branching execution method for fractional LP solutions */
   static SCIP_DECL_BRANCHEXECLP(branchExeclpDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;
      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_execlp(scip, branchrule, allowaddcons, result));

      return SCIP_OKAY;
   }

   /** branching execution method for external candidates */
   static SCIP_DECL_BRANCHEXECEXT(branchExecextDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_execext(scip, branchrule, allowaddcons, result));

      return SCIP_OKAY;
   }

   /** branching execution method for not completely fixed pseudo solutions */
   static SCIP_DECL_BRANCHEXECPS(branchExecpsDynamicBranching)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_execps(scip, branchrule, allowaddcons, result));

      return SCIP_OKAY;
   }
}


/*
 *                                           Callback methods of branching rule
 * --------------------------------------------------------------------------------------------------------------
 */


static SCIP_RETCODE printNodeRootPath(
    SCIP *scip,      /**< SCIP data structure */
    SCIP_NODE *node, /**< node data */
    FILE *file       /**< output file (or NULL for standard output) */
)
{
   SCIP_VAR **branchvars;      /* array of variables on which the branchings has been performed in all ancestors */
   double *branchbounds;    /* array of bounds which the branchings in all ancestors set */
   SCIP_BOUNDTYPE *boundtypes; /* array of boundtypes which the branchings in all ancestors set */
   int *nodeswitches;          /* marks, where in the arrays the branching decisions of the next node on the path start
                                              * branchings performed at the parent of node always start at position 0. For single variable branching,
                                              * nodeswitches[i] = i holds */
   int nbranchvars;            /* number of variables on which branchings have been performed in all ancestors
                                              *   if this is larger than the array size, arrays should be reallocated and method should be called again */
   int branchvarssize;         /* available slots in arrays */
   int nnodes;                 /* number of nodes in the nodeswitch array */
   int nodeswitchsize;         /* available slots in node switch array */

   branchvarssize = SCIPnodeGetDepth(node);
   nodeswitchsize = branchvarssize;

   /* memory allocation */
   SCIP_CALL(SCIPallocBufferArray(scip, &branchvars, branchvarssize));
   SCIP_CALL(SCIPallocBufferArray(scip, &branchbounds, branchvarssize));
   SCIP_CALL(SCIPallocBufferArray(scip, &boundtypes, branchvarssize));
   SCIP_CALL(SCIPallocBufferArray(scip, &nodeswitches, nodeswitchsize));

   SCIPnodeGetAncestorBranchingPath(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize, nodeswitches, &nnodes, nodeswitchsize);
   /* if the arrays were to small, we have to reallocate them and recall SCIPnodeGetAncestorBranchingPath */
   if (nbranchvars > branchvarssize || nnodes > nodeswitchsize)
   {
      branchvarssize = nbranchvars;
      nodeswitchsize = nnodes;
      
      /* memory reallocation */
      SCIP_CALL(SCIPreallocBufferArray(scip, &branchvars, branchvarssize));
      SCIP_CALL(SCIPreallocBufferArray(scip, &branchbounds, branchvarssize));
      SCIP_CALL(SCIPreallocBufferArray(scip, &boundtypes, branchvarssize));
      SCIP_CALL(SCIPreallocBufferArray(scip, &nodeswitches, nodeswitchsize));

      SCIPnodeGetAncestorBranchingPath(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize, nodeswitches, &nnodes, nodeswitchsize);
      assert(nbranchvars == branchvarssize);
   }

   /* we only want to create output, if branchings were performed */
   if (nbranchvars >= 1)
   {
      int i;
      int j;

      /* print all nodes, starting from the root, which is last in the arrays */
      for (j = nnodes - 1; j >= 0; --j)
      {
         int end;
         if (j == nnodes - 1)
            end = nbranchvars;
         else
            end = nodeswitches[j + 1];

         for (i = nodeswitches[j]; i < end; ++i)
         {
            if (i > nodeswitches[j])
               printf(" AND ");
            printf("<%s> %s %.1f", SCIPvarGetName(branchvars[i]), boundtypes[i] == SCIP_BOUNDTYPE_LOWER ? ">=" : "<=", branchbounds[i]);
         }
         printf("\n");
         if (j > 0)
         {
            if (nodeswitches[j] - nodeswitches[j - 1] != 1)
               printf(" |\n |\n");
            else if (boundtypes[i - 1] == SCIP_BOUNDTYPE_LOWER)
               printf("\\ \n \\\n");
            else
               printf(" /\n/ \n");
         }
      }
   }

   /* free all local memory */
   SCIPfreeBufferArray(scip, &nodeswitches);
   SCIPfreeBufferArray(scip, &boundtypes);
   SCIPfreeBufferArray(scip, &branchbounds);
   SCIPfreeBufferArray(scip, &branchvars);

   return SCIP_OKAY;
}