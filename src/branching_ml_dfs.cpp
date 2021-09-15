#include "branching_ml_dfs.hpp"
#include "utils.hpp"
#include <boost/algorithm/string.hpp> 
#include <math.h> 

/** branching rule data */
struct SCIP_BranchruleData
{
   scip::ObjBranchrule *objbranchrule; /**< branching rule object */
   SCIP_Bool deleteobject;             /**< should the branching rule object be deleted when branching rule is freed? */
};

COML::Branching_dfs::Branching_dfs(
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


SCIP_DECL_BRANCHEXECLP(COML::Branching_dfs::scip_execlp)
{
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

   if (nlpcands == 1)
   {
      bestcand = 0;
   }
   else
   {
      double cands_ml_score[nlpcands];

      const char* var_name;
      int var_idx;
      char var_short_name[10];
      int prefix_remove =  gconf.prefix_size_remove + 2;
      if (gconf.policy==Policy::ML_DFS_HEUR_SCORE1_GCN ||
         gconf.policy==Policy::ML_DFS_HEUR_SCORE1_LR ||
            gconf.policy == Policy::SCIP_DEF_PBDFS)
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
      // else if (gconf.policy==Policy::ML_DFS_HEUR_SCORE3_GCN)
      // {
      //    for (int i = 0; i < nlpcands; i++)
      //    {
      //       var_name = SCIPvarGetName(lpcands[i]);
      //       strcpy(var_short_name, var_name + prefix_remove);

      //       if (gconf.var_mapper.find(var_short_name) == gconf.var_mapper.end())
      //       {
      //          cands_ml_score[i] = 0;
      //       }
      //       else
      //       {
      //          var_idx = gconf.var_mapper[var_short_name];
      //          cands_ml_score[i] = 1-gconf.ml_scores1[var_idx];
      //       }
      //    }
      // }
      else if (gconf.policy==Policy::ML_DFS_HEUR_SCORE2_GCN ||
                  gconf.policy==Policy::ML_DFS_HEUR_SCORE2_LR)
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
   }
   
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

   return SCIP_OKAY;
};



/*
 *                                         Callback methods of SCIP
 * --------------------------------------------------------------------------------------------------------------
 */

extern "C"
{

   /** copy method for branchrule plugins (called when SCIP copies plugins) */
   static SCIP_DECL_BRANCHCOPY(branchCopyBranchingMLDFS)
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
         SCIP_CALL(COML::SCIPincludeObjBranchrule_MLDFS(scip, newobjbranchrule, TRUE));
      }
      return SCIP_OKAY;
   }

   /** destructor of branching rule to free user data (called when SCIP is exiting) */
   static SCIP_DECL_BRANCHFREE(branchFreeBranchingMLDFS)
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
   static SCIP_DECL_BRANCHINIT(branchInitBranchingMLDFS)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      assert(branchruledata->objbranchrule->scip_ == scip);

      return SCIP_OKAY;
   }

   /** deinitialization method of branching rule (called before transformed problem is freed) */
   static SCIP_DECL_BRANCHEXIT(branchExitBranchingMLDFS)
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
   static SCIP_DECL_BRANCHINITSOL(branchInitsolBranchingMLDFS)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata;

      branchruledata = SCIPbranchruleGetData(branchrule);
      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);

      return SCIP_OKAY;
   }

   /** solving process deinitialization method of branching rule (called before branch and bound process data is freed) */
   static SCIP_DECL_BRANCHEXITSOL(branchExitsolBranchingMLDFS)
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
   static SCIP_DECL_BRANCHEXECLP(branchExeclpBranchingMLDFS)
   { /*lint --e{715}*/
      SCIP_BRANCHRULEDATA *branchruledata = NULL;
      branchruledata = SCIPbranchruleGetData(branchrule);

      assert(branchruledata != NULL);
      assert(branchruledata->objbranchrule != NULL);
      /* call virtual method of branchrule object */
      SCIP_CALL(branchruledata->objbranchrule->scip_execlp(scip, branchrule, allowaddcons, result));

      return SCIP_OKAY;
   }

   /** branching execution method for external candidates */
   static SCIP_DECL_BRANCHEXECEXT(branchExecextBranchingMLDFS)
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
   static SCIP_DECL_BRANCHEXECPS(branchExecpsBranchingMLDFS)
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



/** creates the branching rule for the given branching rule object and includes it in SCIP */
SCIP_RETCODE COML::SCIPincludeObjBranchrule_MLDFS(
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
                                 branchCopyBranchingMLDFS,
                                 branchFreeBranchingMLDFS, branchInitBranchingMLDFS, branchExitBranchingMLDFS, branchInitsolBranchingMLDFS, branchExitsolBranchingMLDFS,
                                 branchExeclpBranchingMLDFS, branchExecextBranchingMLDFS, branchExecpsBranchingMLDFS,
                                 branchruledata)); /*lint !e429*/

   return SCIP_OKAY; /*lint !e429*/
}