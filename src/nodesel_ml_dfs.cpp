#include <cassert>
#include "nodesel_ml_dfs.hpp"
#include "global_config.hpp"
#include <scip/type_var.h>
#include <scip/struct_var.h>
#include <scip/type_tree.h>
#include <scip/struct_tree.h>
#include <scip/type_lp.h>
/** node selector data */
struct SCIP_NodeselData
{
   scip::ObjNodesel*     objnodesel;         /**< node selector object */
   SCIP_Bool             deleteobject;       /**< should the node selector object be deleted when node selector is freed? */
};


SCIP_DECL_NODESELSELECT(COML::Nodesel_ML_DFS::scip_select)
{
   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), scip_name_) == 0);
   assert(scip != NULL);
   assert(selnode != NULL);
   *selnode = SCIPgetBestNode(scip);
   return SCIP_OKAY;
};

SCIP_DECL_NODESELCOMP(COML::Nodesel_ML_DFS::scip_comp)
{
   int depth1;
   int depth2;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), scip_name_) == 0);
   assert(scip != NULL);

   depth1 = SCIPnodeGetDepth(node1);
   depth2 = SCIPnodeGetDepth(node2);
   if( depth1 > depth2 )
      return -1;
   else if( depth1 < depth2 )
      return +1;
   else
   {  
      unsigned int node1_bd_type = node1->domchg->domchgbound.boundchgs[0].boundtype;
      unsigned int node2_bd_type = node2->domchg->domchgbound.boundchgs[0].boundtype;
      if (gconf.policy==Policy::ML_DFS_HEUR_SCORE1_GCN ||
            gconf.policy==Policy::ML_DFS_EXACT_SCORE1_GCN || 
            gconf.policy==Policy::ML_DFS_HEUR_SCORE1_LR || 
            gconf.policy == Policy::SCIP_DEF_PBDFS)
      {

         if (node1_bd_type == SCIP_BOUNDTYPE_LOWER)
            return -1;
         else if(node2_bd_type == SCIP_BOUNDTYPE_LOWER)
            return +1;
         else
            return 0;
      } 
      else if (gconf.policy==Policy::ML_DFS_EXACT_SCORE3_GCN)
      {

         if (node1_bd_type == SCIP_BOUNDTYPE_LOWER)
            return +1;
         else if(node2_bd_type == SCIP_BOUNDTYPE_LOWER)
            return -1;
         else
            return 0;
      }
      else if (gconf.policy==Policy::ML_DFS_HEUR_SCORE2_GCN ||
               gconf.policy==Policy::ML_DFS_EXACT_SCORE2_GCN ||
               gconf.policy==Policy::ML_DFS_HEUR_SCORE2_LR)
      {
         const char* var_name;
         int var_idx;
         char var_short_name[10];
         int prefix_remove;
         if (gconf.policy==Policy::ML_DFS_EXACT_SCORE2_GCN)
            prefix_remove =  gconf.prefix_size_remove;
         else
            prefix_remove =  gconf.prefix_size_remove + 2;
         var_name = SCIPvarGetName(node1->domchg->domchgbound.boundchgs[0].var);
         strcpy(var_short_name, var_name + prefix_remove);
         var_idx = gconf.var_mapper[var_short_name];
         if (gconf.ml_scores1[var_idx] > 0.5)
         {
            if (node1_bd_type == SCIP_BOUNDTYPE_LOWER)
               return -1;
            else if(node2_bd_type == SCIP_BOUNDTYPE_LOWER)
               return +1;
            else
               return 0;
         }
         else
         {
            if (node1_bd_type == SCIP_BOUNDTYPE_LOWER)
               return +1;
            else if(node2_bd_type == SCIP_BOUNDTYPE_LOWER)
               return -1;
         }
      }
   }
};








/*
 * Callback methods of node selector
 */

extern "C"
{

/** copy method for node selector plugins (called when SCIP copies plugins) */
static
SCIP_DECL_NODESELCOPY(nodeselCopyObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;
   
   assert(scip != NULL);
   
   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);
   assert(nodeseldata->objnodesel->scip_ != scip);

   if( nodeseldata->objnodesel->iscloneable() )
   {
      scip::ObjNodesel* newobjnodesel;
      newobjnodesel = dynamic_cast<scip::ObjNodesel*> (nodeseldata->objnodesel->clone(scip));

      /* call include method of node selector object */
      SCIP_CALL( SCIPincludeObjNodesel(scip, newobjnodesel, TRUE) );
   }

   return SCIP_OKAY;
}

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELFREE(nodeselFreeObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);
   assert(nodeseldata->objnodesel->scip_ == scip);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_free(scip, nodesel) );

   /* free nodesel object */
   if( nodeseldata->deleteobject ){
      delete nodeseldata->objnodesel;
      nodeseldata->objnodesel = NULL;
   }
   /* free nodesel data */
   delete nodeseldata;
   SCIPnodeselSetData(nodesel, NULL); /*lint !e64*/
   
   return SCIP_OKAY;
}


/** initialization method of node selector (called after problem was transformed) */
static
SCIP_DECL_NODESELINIT(nodeselInitObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);
   assert(nodeseldata->objnodesel->scip_ == scip);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_init(scip, nodesel) );

   return SCIP_OKAY;
}


/** deinitialization method of node selector (called before transformed problem is freed) */
static
SCIP_DECL_NODESELEXIT(nodeselExitObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_exit(scip, nodesel) );

   return SCIP_OKAY;
}


/** solving process initialization method of node selector (called when branch and bound process is about to begin) */
static
SCIP_DECL_NODESELINITSOL(nodeselInitsolObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_initsol(scip, nodesel) );

   return SCIP_OKAY;
}


/** solving process deinitialization method of node selector (called before branch and bound process data is freed) */
static
SCIP_DECL_NODESELEXITSOL(nodeselExitsolObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_exitsol(scip, nodesel) );

   return SCIP_OKAY;
}


/** node selection method of node selector */
static
SCIP_DECL_NODESELSELECT(nodeselSelectObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);

   /* call virtual method of nodesel object */
   SCIP_CALL( nodeseldata->objnodesel->scip_select(scip, nodesel, selnode) );

   return SCIP_OKAY;
}


/** node comparison method of node selector */
static
SCIP_DECL_NODESELCOMP(nodeselCompObj)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   assert(nodeseldata->objnodesel != NULL);

   /* call virtual method of nodesel object */
   return nodeseldata->objnodesel->scip_comp(scip, nodesel, node1, node2);
}
}



/*
 * node selector specific interface methods
 */

/** creates the node selector for the given node selector object and includes it in SCIP */
SCIP_RETCODE SCIPincludeObjNodesel_MLDFS(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjNodesel*     objnodesel,         /**< node selector object */
   SCIP_Bool             deleteobject        /**< should the node selector object be deleted when node selector is freed? */
   )
{
   SCIP_NODESELDATA* nodeseldata;

   assert(scip != NULL);
   assert(objnodesel != NULL);

   /* create node selector data */
   nodeseldata = new SCIP_NODESELDATA;
   nodeseldata->objnodesel = objnodesel;
   nodeseldata->deleteobject = deleteobject;

   /* include node selector */
   SCIP_CALL( SCIPincludeNodesel(scip, objnodesel->scip_name_, objnodesel->scip_desc_,
         objnodesel->scip_stdpriority_, objnodesel->scip_memsavepriority_,
         nodeselCopyObj,
         nodeselFreeObj, nodeselInitObj, nodeselExitObj,
         nodeselInitsolObj, nodeselExitsolObj, nodeselSelectObj, nodeselCompObj,
         nodeseldata) ); /*lint !e429*/

   return SCIP_OKAY; /*lint !e429*/
}
