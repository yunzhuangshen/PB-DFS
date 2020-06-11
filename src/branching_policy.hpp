#ifndef __SCIP_BRANCH_DYNAMIC_H__
#define __SCIP_BRANCH_DYNAMIC_H__
#include <vector>
#include <utility>
#include <cassert>
#include <unordered_map>
#include <chrono>
#include <random>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <string>


#include <scip/scip.h>
#include <objscip/objscip.h>
#include <scip/struct_var.h>
#include <scip/type_stat.h>
#include <scip/struct_stat.h>
#include <scip/type_scip.h>
#include <scip/struct_scip.h>
#include <scip/struct_tree.h>
#include <scip/tree.h>
#include <scip/struct_var.h>
#include <scip/var.h>
#include "global_config.hpp"
#include <scip/scip_mem.h>
#include <scip/set.h>
#include <scip/struct_mem.h>
#include <scip/visual.h>
#include <scip/cons_linear.h>
#include <scip/type_cons.h>
#include <scip/struct_cons.h>
#include <scip/struct_nodesel.h>
#include <boost/algorithm/string.hpp> 
#include <math.h> 
#include <scip/branch_relpscost.h>
namespace COML{

EXTERN
SCIP_RETCODE SCIPincludeObjBranchrule_DB(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjBranchrule*  objbranchrule,      /**< branching rule object */
   SCIP_Bool             deleteobject        /**< should the branching rule object be deleted when branching rule is freed? */
   );


class Branching : public scip::ObjBranchrule
{
public:
      int num_vars;

      Branching(
      // ----------------------------------------------  SCIP parameters  --------------------------------------------
      SCIP*              scip,               /**< SCIP data structure */
      const char*        name,               /**< name of branching rule */
      const char*        desc,               /**< description of branching rule */
      int                priority,           /**< priority of the branching rule */
      int                maxdepth,           /**< maximal depth level, up to which this branching rule should be used (or -1) */
      SCIP_Real          maxbounddist        /**< maximal relative distance from current node's dual bound to primal bound
                                              *   compared to best node's dual bound for applying branching rule
                                              *   (0.0: only on current best node, 1.0: on all nodes) */
      // ----------------------------------------------  ml parameters  --------------------------------------------                       
      );

      ~Branching();

      
      void record_solving_curves(SCIP* scip);

   /** branching execution method for fractional LP solutions
    *
    *  @see SCIP_DECL_BRANCHEXECLP(x) in @ref type_branch.h
    */
   SCIP_DECL_BRANCHEXECLP(scip_execlp) override;

   SCIP_DECL_BRANCHINIT(scip_init) override;
};
}

#endif