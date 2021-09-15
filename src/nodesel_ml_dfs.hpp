#ifndef __NODESEL_ML_DFS___
#define __NODESEL_ML_DFS___
#include <scip/scip.h>
#include <objscip/objscip.h>

namespace COML{

class Nodesel_ML_DFS: public scip::ObjNodesel
{
public:

   /** default constructor */
   Nodesel_ML_DFS(
      SCIP*              scip,               /**< SCIP data structure */
      const char*        name,               /**< name of node selector */
      const char*        desc,               /**< description of node selector */
      int                stdpriority,        /**< priority of the node selector in standard mode */
      int                memsavepriority     /**< priority of the node selector in memory saving mode */
      ) : ObjNodesel (scip, name, desc, stdpriority, memsavepriority){};

   SCIP_DECL_NODESELSELECT(scip_select) override;
   SCIP_DECL_NODESELCOMP(scip_comp) override;

};
};

extern
SCIP_RETCODE SCIPincludeObjNodesel_MLDFS(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjNodesel*     objnodesel,         /**< node selector object */
   SCIP_Bool             deleteobject        /**< should the node selector object be deleted when node selector is freed? */
   );


#endif