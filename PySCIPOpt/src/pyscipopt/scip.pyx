##@file scip.pyx
#@brief holding functions in python that reference the SCIP public functions included in scip.pxd
import weakref
from os.path import abspath
from os.path import splitext
import sys
import warnings
import numpy as np
import math
cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT
from libc.stdio cimport printf

include "expr.pxi"
include "lp.pxi"
include "benders.pxi"
include "benderscut.pxi"
include "branchrule.pxi"
include "conshdlr.pxi"
include "event.pxi"
include "heuristic.pxi"
include "presol.pxi"
include "pricer.pxi"
include "propagator.pxi"
include "sepa.pxi"
include "relax.pxi"
include "nodesel.pxi"

# recommended SCIP version; major version is required
MAJOR = 6
MINOR = 0
PATCH = 0

# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

# Mapping the SCIP_RESULT enum to a python class
# This is required to return SCIP_RESULT in the python code
# In __init__.py this is imported as SCIP_RESULT to keep the
# original naming scheme using capital letters
cdef class PY_SCIP_RESULT:
    DIDNOTRUN   = SCIP_DIDNOTRUN
    DELAYED     = SCIP_DELAYED
    DIDNOTFIND  = SCIP_DIDNOTFIND
    FEASIBLE    = SCIP_FEASIBLE
    INFEASIBLE  = SCIP_INFEASIBLE
    UNBOUNDED   = SCIP_UNBOUNDED
    CUTOFF      = SCIP_CUTOFF
    SEPARATED   = SCIP_SEPARATED
    NEWROUND    = SCIP_NEWROUND
    REDUCEDDOM  = SCIP_REDUCEDDOM
    CONSADDED   = SCIP_CONSADDED
    CONSCHANGED = SCIP_CONSCHANGED
    BRANCHED    = SCIP_BRANCHED
    SOLVELP     = SCIP_SOLVELP
    FOUNDSOL    = SCIP_FOUNDSOL
    SUSPENDED   = SCIP_SUSPENDED
    SUCCESS     = SCIP_SUCCESS

cdef class PY_SCIP_PARAMSETTING:
    DEFAULT     = SCIP_PARAMSETTING_DEFAULT
    AGGRESSIVE  = SCIP_PARAMSETTING_AGGRESSIVE
    FAST        = SCIP_PARAMSETTING_FAST
    OFF         = SCIP_PARAMSETTING_OFF

cdef class PY_SCIP_PARAMEMPHASIS:
    DEFAULT      = SCIP_PARAMEMPHASIS_DEFAULT
    CPSOLVER     = SCIP_PARAMEMPHASIS_CPSOLVER
    EASYCIP      = SCIP_PARAMEMPHASIS_EASYCIP
    FEASIBILITY  = SCIP_PARAMEMPHASIS_FEASIBILITY
    HARDLP       = SCIP_PARAMEMPHASIS_HARDLP
    OPTIMALITY   = SCIP_PARAMEMPHASIS_OPTIMALITY
    COUNTER      = SCIP_PARAMEMPHASIS_COUNTER
    PHASEFEAS    = SCIP_PARAMEMPHASIS_PHASEFEAS
    PHASEIMPROVE = SCIP_PARAMEMPHASIS_PHASEIMPROVE
    PHASEPROOF   = SCIP_PARAMEMPHASIS_PHASEPROOF

cdef class PY_SCIP_STATUS:
    UNKNOWN        = SCIP_STATUS_UNKNOWN
    USERINTERRUPT  = SCIP_STATUS_USERINTERRUPT
    NODELIMIT      = SCIP_STATUS_NODELIMIT
    TOTALNODELIMIT = SCIP_STATUS_TOTALNODELIMIT
    STALLNODELIMIT = SCIP_STATUS_STALLNODELIMIT
    TIMELIMIT      = SCIP_STATUS_TIMELIMIT
    MEMLIMIT       = SCIP_STATUS_MEMLIMIT
    GAPLIMIT       = SCIP_STATUS_GAPLIMIT
    SOLLIMIT       = SCIP_STATUS_SOLLIMIT
    BESTSOLLIMIT   = SCIP_STATUS_BESTSOLLIMIT
    RESTARTLIMIT   = SCIP_STATUS_RESTARTLIMIT
    OPTIMAL        = SCIP_STATUS_OPTIMAL
    INFEASIBLE     = SCIP_STATUS_INFEASIBLE
    UNBOUNDED      = SCIP_STATUS_UNBOUNDED
    INFORUNBD      = SCIP_STATUS_INFORUNBD

cdef class PY_SCIP_STAGE:
    INIT         = SCIP_STAGE_INIT
    PROBLEM      = SCIP_STAGE_PROBLEM
    TRANSFORMING = SCIP_STAGE_TRANSFORMING
    TRANSFORMED  = SCIP_STAGE_TRANSFORMED
    INITPRESOLVE = SCIP_STAGE_INITPRESOLVE
    PRESOLVING   = SCIP_STAGE_PRESOLVING
    EXITPRESOLVE = SCIP_STAGE_EXITPRESOLVE
    PRESOLVED    = SCIP_STAGE_PRESOLVED
    INITSOLVE    = SCIP_STAGE_INITSOLVE
    SOLVING      = SCIP_STAGE_SOLVING
    SOLVED       = SCIP_STAGE_SOLVED
    EXITSOLVE    = SCIP_STAGE_EXITSOLVE
    FREETRANS    = SCIP_STAGE_FREETRANS
    FREE         = SCIP_STAGE_FREE

cdef class PY_SCIP_NODETYPE:
    FOCUSNODE   = SCIP_NODETYPE_FOCUSNODE
    PROBINGNODE = SCIP_NODETYPE_PROBINGNODE
    SIBLING     = SCIP_NODETYPE_SIBLING
    CHILD       = SCIP_NODETYPE_CHILD
    LEAF        = SCIP_NODETYPE_LEAF
    DEADEND     = SCIP_NODETYPE_DEADEND
    JUNCTION    = SCIP_NODETYPE_JUNCTION
    PSEUDOFORK  = SCIP_NODETYPE_PSEUDOFORK
    FORK        = SCIP_NODETYPE_FORK
    SUBROOT     = SCIP_NODETYPE_SUBROOT
    REFOCUSNODE = SCIP_NODETYPE_REFOCUSNODE


cdef class PY_SCIP_PROPTIMING:
    BEFORELP     = SCIP_PROPTIMING_BEFORELP
    DURINGLPLOOP = SCIP_PROPTIMING_DURINGLPLOOP
    AFTERLPLOOP  = SCIP_PROPTIMING_AFTERLPLOOP
    AFTERLPNODE  = SCIP_PROPTIMING_AFTERLPNODE

cdef class PY_SCIP_PRESOLTIMING:
    NONE       = SCIP_PRESOLTIMING_NONE
    FAST       = SCIP_PRESOLTIMING_FAST
    MEDIUM     = SCIP_PRESOLTIMING_MEDIUM
    EXHAUSTIVE = SCIP_PRESOLTIMING_EXHAUSTIVE

cdef class PY_SCIP_HEURTIMING:
    BEFORENODE        = SCIP_HEURTIMING_BEFORENODE
    DURINGLPLOOP      = SCIP_HEURTIMING_DURINGLPLOOP
    AFTERLPLOOP       = SCIP_HEURTIMING_AFTERLPLOOP
    AFTERLPNODE       = SCIP_HEURTIMING_AFTERLPNODE
    AFTERPSEUDONODE   = SCIP_HEURTIMING_AFTERPSEUDONODE
    AFTERLPPLUNGE     = SCIP_HEURTIMING_AFTERLPPLUNGE
    AFTERPSEUDOPLUNGE = SCIP_HEURTIMING_AFTERPSEUDOPLUNGE
    DURINGPRICINGLOOP = SCIP_HEURTIMING_DURINGPRICINGLOOP
    BEFOREPRESOL      = SCIP_HEURTIMING_BEFOREPRESOL
    DURINGPRESOLLOOP  = SCIP_HEURTIMING_DURINGPRESOLLOOP
    AFTERPROPLOOP     = SCIP_HEURTIMING_AFTERPROPLOOP

cdef class PY_SCIP_EVENTTYPE:
    DISABLED        = SCIP_EVENTTYPE_DISABLED
    VARADDED        = SCIP_EVENTTYPE_VARADDED
    VARDELETED      = SCIP_EVENTTYPE_VARDELETED
    VARFIXED        = SCIP_EVENTTYPE_VARFIXED
    VARUNLOCKED     = SCIP_EVENTTYPE_VARUNLOCKED
    OBJCHANGED      = SCIP_EVENTTYPE_OBJCHANGED
    GLBCHANGED      = SCIP_EVENTTYPE_GLBCHANGED
    GUBCHANGED      = SCIP_EVENTTYPE_GUBCHANGED
    LBTIGHTENED     = SCIP_EVENTTYPE_LBTIGHTENED
    LBRELAXED       = SCIP_EVENTTYPE_LBRELAXED
    UBTIGHTENED     = SCIP_EVENTTYPE_UBTIGHTENED
    UBRELAXED       = SCIP_EVENTTYPE_UBRELAXED
    GHOLEADDED      = SCIP_EVENTTYPE_GHOLEADDED
    GHOLEREMOVED    = SCIP_EVENTTYPE_GHOLEREMOVED
    LHOLEADDED      = SCIP_EVENTTYPE_LHOLEADDED
    LHOLEREMOVED    = SCIP_EVENTTYPE_LHOLEREMOVED
    IMPLADDED       = SCIP_EVENTTYPE_IMPLADDED
    PRESOLVEROUND   = SCIP_EVENTTYPE_PRESOLVEROUND
    NODEFOCUSED     = SCIP_EVENTTYPE_NODEFOCUSED
    NODEFEASIBLE    = SCIP_EVENTTYPE_NODEFEASIBLE
    NODEINFEASIBLE  = SCIP_EVENTTYPE_NODEINFEASIBLE
    NODEBRANCHED    = SCIP_EVENTTYPE_NODEBRANCHED
    FIRSTLPSOLVED   = SCIP_EVENTTYPE_FIRSTLPSOLVED
    LPSOLVED        = SCIP_EVENTTYPE_LPSOLVED
    LPEVENT         = SCIP_EVENTTYPE_LPEVENT
    POORSOLFOUND    = SCIP_EVENTTYPE_POORSOLFOUND
    BESTSOLFOUND    = SCIP_EVENTTYPE_BESTSOLFOUND
    ROWADDEDSEPA    = SCIP_EVENTTYPE_ROWADDEDSEPA
    ROWDELETEDSEPA  = SCIP_EVENTTYPE_ROWDELETEDSEPA
    ROWADDEDLP      = SCIP_EVENTTYPE_ROWADDEDLP
    ROWDELETEDLP    = SCIP_EVENTTYPE_ROWDELETEDLP
    ROWCOEFCHANGED  = SCIP_EVENTTYPE_ROWCOEFCHANGED
    ROWCONSTCHANGED = SCIP_EVENTTYPE_ROWCONSTCHANGED
    ROWSIDECHANGED  = SCIP_EVENTTYPE_ROWSIDECHANGED
    SYNC            = SCIP_EVENTTYPE_SYNC

cdef class PY_SCIP_LPSOLSTAT:
    NOTSOLVED    = SCIP_LPSOLSTAT_NOTSOLVED
    OPTIMAL      = SCIP_LPSOLSTAT_OPTIMAL
    INFEASIBLE   = SCIP_LPSOLSTAT_INFEASIBLE
    UNBOUNDEDRAY = SCIP_LPSOLSTAT_UNBOUNDEDRAY
    OBJLIMIT     = SCIP_LPSOLSTAT_OBJLIMIT
    ITERLIMIT    = SCIP_LPSOLSTAT_ITERLIMIT
    TIMELIMIT    = SCIP_LPSOLSTAT_TIMELIMIT
    ERROR        = SCIP_LPSOLSTAT_ERROR

cdef class PY_SCIP_BRANCHDIR:
    DOWNWARDS = SCIP_BRANCHDIR_DOWNWARDS
    UPWARDS   = SCIP_BRANCHDIR_UPWARDS
    FIXED     = SCIP_BRANCHDIR_FIXED
    AUTO      = SCIP_BRANCHDIR_AUTO

cdef class PY_SCIP_BENDERSENFOTYPE:
    LP     = SCIP_BENDERSENFOTYPE_LP
    RELAX  = SCIP_BENDERSENFOTYPE_RELAX
    PSEUDO = SCIP_BENDERSENFOTYPE_PSEUDO
    CHECK  = SCIP_BENDERSENFOTYPE_CHECK


def PY_SCIP_CALL(SCIP_RETCODE rc):
    if rc == SCIP_OKAY:
        pass
    elif rc == SCIP_ERROR:
        raise Exception('SCIP: unspecified error!')
    elif rc == SCIP_NOMEMORY:
        raise MemoryError('SCIP: insufficient memory error!')
    elif rc == SCIP_READERROR:
        raise IOError('SCIP: read error!')
    elif rc == SCIP_WRITEERROR:
        raise IOError('SCIP: write error!')
    elif rc == SCIP_NOFILE:
        raise IOError('SCIP: file not found error!')
    elif rc == SCIP_FILECREATEERROR:
        raise IOError('SCIP: cannot create file!')
    elif rc == SCIP_LPERROR:
        raise Exception('SCIP: error in LP solver!')
    elif rc == SCIP_NOPROBLEM:
        raise Exception('SCIP: no problem exists!')
    elif rc == SCIP_INVALIDCALL:
        raise Exception('SCIP: method cannot be called at this time'
                            + ' in solution process!')
    elif rc == SCIP_INVALIDDATA:
        raise Exception('SCIP: error in input data!')
    elif rc == SCIP_INVALIDRESULT:
        raise Exception('SCIP: method returned an invalid result code!')
    elif rc == SCIP_PLUGINNOTFOUND:
        raise Exception('SCIP: a required plugin was not found !')
    elif rc == SCIP_PARAMETERUNKNOWN:
        raise KeyError('SCIP: the parameter with the given name was not found!')
    elif rc == SCIP_PARAMETERWRONGTYPE:
        raise LookupError('SCIP: the parameter is not of the expected type!')
    elif rc == SCIP_PARAMETERWRONGVAL:
        raise ValueError('SCIP: the value is invalid for the given parameter!')
    elif rc == SCIP_KEYALREADYEXISTING:
        raise KeyError('SCIP: the given key is already existing in table!')
    elif rc == SCIP_MAXDEPTHLEVEL:
        raise Exception('SCIP: maximal branching depth level exceeded!')
    else:
        raise Exception('SCIP: unknown return code!')

cdef class Event:
    cdef SCIP_EVENT* event
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_EVENT* scip_event):
        event = Event()
        event.event = scip_event
        return event

    def getType(self):
        """gets type of event"""
        return SCIPeventGetType(self.event)

    def __repr__(self):
        return self.getType()

    def getNewBound(self):
        """gets new bound for a bound change event"""
        return SCIPeventGetNewbound(self.event)

    def getOldBound(self):
        """gets old bound for a bound change event"""
        return SCIPeventGetOldbound(self.event)

    def getVar(self):
        """gets variable for a variable event (var added, var deleted, var fixed, objective value or domain change, domain hole added or removed)"""
        cdef SCIP_VAR* var = SCIPeventGetVar(self.event)
        return Variable.create(var)

    def getNode(self):
        """gets node for a node or LP event"""
        cdef SCIP_NODE* node = SCIPeventGetNode(self.event)
        return Node.create(node)

cdef class Column:
    """Base class holding a pointer to corresponding SCIP_COL"""
    cdef SCIP_COL* scip_col
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_COL* scipcol):
        col = Column()
        col.scip_col = scipcol
        return col

    def getLPPos(self):
        """gets position of column in current LP, or -1 if it is not in LP"""
        return SCIPcolGetLPPos(self.scip_col)

    def getBasisStatus(self):
        """gets the basis status of a column in the LP solution, Note: returns basis status `zero` for columns not in the current SCIP LP"""
        cdef SCIP_BASESTAT stat = SCIPcolGetBasisStatus(self.scip_col)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            return "zero"
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """returns whether the associated variable is of integral type (binary, integer, implicit integer)"""
        return SCIPcolIsIntegral(self.scip_col)

    def getVar(self):
        """gets variable this column represents"""
        cdef SCIP_VAR* var = SCIPcolGetVar(self.scip_col)
        return Variable.create(var)

    def getPrimsol(self):
        """gets the primal LP solution of a column"""
        return SCIPcolGetPrimsol(self.scip_col)

    def getLb(self):
        """gets lower bound of column"""
        return SCIPcolGetLb(self.scip_col)

    def getUb(self):
        """gets upper bound of column"""
        return SCIPcolGetUb(self.scip_col)

cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW"""
    cdef SCIP_ROW* scip_row
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_ROW* sciprow):
        row = Row()
        row.scip_row = sciprow
        return row

    def getLhs(self):
        """returns the left hand side of row"""
        return SCIProwGetLhs(self.scip_row)

    def getRhs(self):
        """returns the right hand side of row"""
        return SCIProwGetRhs(self.scip_row)

    def getConstant(self):
        """gets constant shift of row"""
        return SCIProwGetConstant(self.scip_row)

    def getLPPos(self):
        """gets position of row in current LP, or -1 if it is not in LP"""
        return SCIProwGetLPPos(self.scip_row)

    def getBasisStatus(self):
        """gets the basis status of a row in the LP solution, Note: returns basis status `basic` for rows not in the current SCIP LP"""
        cdef SCIP_BASESTAT stat = SCIProwGetBasisStatus(self.scip_row)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            # this shouldn't happen!
            raise Exception('SCIP returned base status zero for a row!')
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """returns TRUE iff the activity of the row (without the row's constant) is always integral in a feasible solution """
        return SCIProwIsIntegral(self.scip_row)

    def isModifiable(self):
        """returns TRUE iff row is modifiable during node processing (subject to column generation) """
        return SCIProwIsModifiable(self.scip_row)

    def getNNonz(self):
        """get number of nonzero entries in row vector"""
        return SCIProwGetNNonz(self.scip_row)

    def getNLPNonz(self):
        """get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP"""
        return SCIProwGetNLPNonz(self.scip_row)

    def getCols(self):
        """gets list with columns of nonzero entries"""
        cdef SCIP_COL** cols = SCIProwGetCols(self.scip_row)
        return [Column.create(cols[i]) for i in range(self.getNNonz())]

    def getVals(self):
        """gets list with coefficients of nonzero entries"""
        cdef SCIP_Real* vals = SCIProwGetVals(self.scip_row)
        return [vals[i] for i in range(self.getNNonz())]

cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL"""
    cdef SCIP_SOL* sol
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_SOL* scip_sol):
        sol = Solution()
        sol.sol = scip_sol
        return sol

cdef class Node:
    """Base class holding a pointer to corresponding SCIP_NODE"""
    cdef SCIP_NODE* scip_node
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_NODE* scipnode):
        node = Node()
        node.scip_node = scipnode
        return node

    def getParent(self):
        """Retrieve parent node."""
        return Node.create(SCIPnodeGetParent(self.scip_node))

    def getNumber(self):
        """Retrieve number of node."""
        return SCIPnodeGetNumber(self.scip_node)

    def getDepth(self):
        """Retrieve depth of node."""
        return SCIPnodeGetDepth(self.scip_node)

    def getType(self):
        """Retrieve type of node."""
        return SCIPnodeGetType(self.scip_node)

    def getLowerbound(self):
        """Retrieve lower bound of node."""
        return SCIPnodeGetLowerbound(self.scip_node)

    def getEstimate(self):
        """Retrieve the estimated value of the best feasible solution in subtree of the node"""
        return SCIPnodeGetEstimate(self.scip_node)

    def getNAddedConss(self):
        """Retrieve number of added constraints at this node"""
        return SCIPnodeGetNAddedConss(self.scip_node)

    def isActive(self):
        """Is the node in the path to the current node?"""
        return SCIPnodeIsActive(self.scip_node)

    def isPropagatedAgain(self):
        """Is the node marked to be propagated again?"""
        return SCIPnodeIsPropagatedAgain(self.scip_node)

    def getBranchInfos(self):
        """Get branching decision of the parent node."""
        domchg = SCIPnodeGetDomchg(self.scip_node)
        nboundchgs = SCIPdomchgGetNBoundchgs(domchg)
        assert nboundchgs == 1
        boundchg = SCIPdomchgGetBoundchg(domchg, 0)

        result = []
        result.append(SCIPboundchgGetNewbound(boundchg))
        result.append(Variable.create(SCIPboundchgGetVar(boundchg)))
        result.append(SCIPboundchgGetBoundchgtype(boundchg))
        result.append(SCIPboundchgGetBoundtype(boundchg))
        result.append(SCIPboundchgIsRedundant(boundchg))
        return result


cdef class Variable(Expr):
    """Is a linear expression and has SCIP_VAR*"""
    cdef SCIP_VAR* scip_var
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_VAR* scipvar):
        var = Variable()
        var.scip_var = scipvar
        Expr.__init__(var, {Term(var) : 1.0})
        return var

    property name:
        def __get__(self):
            cname = bytes( SCIPvarGetName(self.scip_var) )
            return cname.decode('utf-8')

    def ptr(self):
        """ """
        return <size_t>(self.scip_var)

    def __repr__(self):
        return self.name

    def vtype(self):
        """Retrieve the variables type (BINARY, INTEGER or CONTINUOUS)"""
        vartype = SCIPvarGetType(self.scip_var)
        if vartype == SCIP_VARTYPE_BINARY:
            return "BINARY"
        elif vartype == SCIP_VARTYPE_INTEGER:
            return "INTEGER"
        elif vartype == SCIP_VARTYPE_CONTINUOUS or vartype == SCIP_VARTYPE_IMPLINT:
            return "CONTINUOUS"

    def isOriginal(self):
        """Retrieve whether the variable belongs to the original problem"""
        return SCIPvarIsOriginal(self.scip_var)

    def isInLP(self):
        """Retrieve whether the variable is a COLUMN variable that is member of the current LP"""
        return SCIPvarIsInLP(self.scip_var)

    def getIndex(self):
        return SCIPvarGetIndex(self.scip_var)

    def getCol(self):
        """Retrieve column of COLUMN variable"""
        cdef SCIP_COL* scip_col
        scip_col = SCIPvarGetCol(self.scip_var)
        return Column.create(scip_col)

    def getLbOriginal(self):
        """Retrieve original lower bound of variable"""
        return SCIPvarGetLbOriginal(self.scip_var)

    def getUbOriginal(self):
        """Retrieve original upper bound of variable"""
        return SCIPvarGetUbOriginal(self.scip_var)

    def getLbGlobal(self):
        """Retrieve global lower bound of variable"""
        return SCIPvarGetLbGlobal(self.scip_var)

    def getUbGlobal(self):
        """Retrieve global upper bound of variable"""
        return SCIPvarGetUbGlobal(self.scip_var)

    def getLbLocal(self):
        """Retrieve current lower bound of variable"""
        return SCIPvarGetLbLocal(self.scip_var)

    def getUbLocal(self):
        """Retrieve current upper bound of variable"""
        return SCIPvarGetUbLocal(self.scip_var)

    def getObj(self):
        """Retrieve current objective value of variable"""
        return SCIPvarGetObj(self.scip_var)

    def getLPSol(self):
        """Retrieve the current LP solution value of variable"""
        return SCIPvarGetLPSol(self.scip_var)


cdef class Constraint:
    cdef SCIP_CONS* scip_cons
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_CONS* scipcons):
        if scipcons == NULL:
            raise Warning("cannot create Constraint with SCIP_CONS* == NULL")
        cons = Constraint()
        cons.scip_cons = scipcons
        return cons

    property name:
        def __get__(self):
            cname = bytes( SCIPconsGetName(self.scip_cons) )
            return cname.decode('utf-8')

    def __repr__(self):
        return self.name

    def isOriginal(self):
        """Retrieve whether the constraint belongs to the original problem"""
        return SCIPconsIsOriginal(self.scip_cons)

    def isInitial(self):
        """Retrieve True if the relaxation of the constraint should be in the initial LP"""
        return SCIPconsIsInitial(self.scip_cons)

    def isSeparated(self):
        """Retrieve True if constraint should be separated during LP processing"""
        return SCIPconsIsSeparated(self.scip_cons)

    def isEnforced(self):
        """Retrieve True if constraint should be enforced during node processing"""
        return SCIPconsIsEnforced(self.scip_cons)

    def isChecked(self):
        """Retrieve True if constraint should be checked for feasibility"""
        return SCIPconsIsChecked(self.scip_cons)

    def isPropagated(self):
        """Retrieve True if constraint should be propagated during node processing"""
        return SCIPconsIsPropagated(self.scip_cons)

    def isLocal(self):
        """Retrieve True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsLocal(self.scip_cons)

    def isModifiable(self):
        """Retrieve True if constraint is modifiable (subject to column generation)"""
        return SCIPconsIsModifiable(self.scip_cons)

    def isDynamic(self):
        """Retrieve True if constraint is subject to aging"""
        return SCIPconsIsDynamic(self.scip_cons)

    def isRemovable(self):
        """Retrieve True if constraint's relaxation should be removed from the LP due to aging or cleanup"""
        return SCIPconsIsRemovable(self.scip_cons)

    def isStickingAtNode(self):
        """Retrieve True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsStickingAtNode(self.scip_cons)

    def isLinear(self):
        """Retrieve True if constraint is linear"""
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'linear'

    def isQuadratic(self):
        """Retrieve True if constraint is quadratic"""
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'quadratic'


cdef void relayMessage(SCIP_MESSAGEHDLR *messagehdlr, FILE *file, const char *msg):
    sys.stdout.write(msg.decode('UTF-8'))

cdef void relayErrorMessage(void *messagehdlr, FILE *file, const char *msg):
    sys.stderr.write(msg.decode('UTF-8'))

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
##
#@anchor Model
##
cdef class Model:
    cdef SCIP* _scip
    cdef SCIP_Bool* _valid
    # store best solution to get the solution values easier
    cdef Solution _bestSol
    # can be used to store problem data
    cdef public object data
    # make Model weak referentiable
    cdef object __weakref__

    def __init__(self, problemName='model', defaultPlugins=True, Model sourceModel=None, origcopy=False, globalcopy=True, enablepricing=False):
        """
        :param problemName: name of the problem (default 'model')
        :param defaultPlugins: use default plugins? (default True)
        :param sourceModel: create a copy of the given Model instance (default None)
        :param origcopy: whether to call copy or copyOrig (default False)
        :param globalcopy: whether to create a global or a local copy (default True)
        :param enablepricing: whether to enable pricing in copy (default False)
        """
        if self.version() < MAJOR:
            raise Exception("linked SCIP is not compatible to this version of PySCIPOpt - use at least version", MAJOR)
        if self.version() < MAJOR + MINOR/10.0 + PATCH/100.0:
            warnings.warn("linked SCIP {} is not recommended for this version of PySCIPOpt - use version {}.{}.{}".format(self.version(), MAJOR, MINOR, PATCH))
        if sourceModel is None:
            self.create()
            self._bestSol = None
            if defaultPlugins:
                self.includeDefaultPlugins()
            self.createProbBasic(problemName)
        else:
            self.create()
            self._bestSol = <Solution> sourceModel._bestSol
            n = str_conversion(problemName)
            if origcopy:
                PY_SCIP_CALL(SCIPcopyOrig(sourceModel._scip, self._scip, NULL, NULL, n, enablepricing, True, self._valid))
            else:
                PY_SCIP_CALL(SCIPcopy(sourceModel._scip, self._scip, NULL, NULL, n, globalcopy, enablepricing, True, self._valid))

    def __dealloc__(self):
        # call C function directly, because we can no longer call this object's methods, according to
        # http://docs.cython.org/src/reference/extension_types.html#finalization-dealloc
        PY_SCIP_CALL( SCIPfree(&self._scip) )

    def create(self):
        """Create a new SCIP instance"""
        PY_SCIP_CALL(SCIPcreate(&self._scip))

    def includeDefaultPlugins(self):
        """Includes all default plug-ins into SCIP"""
        PY_SCIP_CALL(SCIPincludeDefaultPlugins(self._scip))

    def createProbBasic(self, problemName='model'):
        """Create new problem instance with given name

        :param problemName: name of model or problem (Default value = 'model')

        """
        n = str_conversion(problemName)
        PY_SCIP_CALL(SCIPcreateProbBasic(self._scip, n))

    def freeProb(self):
        """Frees problem and solution process data"""
        PY_SCIP_CALL(SCIPfreeProb(self._scip))

    def freeTransform(self):
        """Frees all solution process data including presolving and transformed problem, only original problem is kept"""
        PY_SCIP_CALL(SCIPfreeTransform(self._scip))

    def version(self):
        """Retrieve SCIP version"""
        return SCIPversion()

    def printVersion(self):
        """Print version, copyright information and compile mode"""
        SCIPprintVersion(self._scip, NULL)

    def getProbName(self):
        """Retrieve problem name"""
        return bytes(SCIPgetProbName(self._scip)).decode('UTF-8')

    def getTotalTime(self):
        """Retrieve the current total SCIP time in seconds, i.e. the total time since the SCIP instance has been created"""
        return SCIPgetTotalTime(self._scip)

    def getSolvingTime(self):
        """Retrieve the current solving time in seconds"""
        return SCIPgetSolvingTime(self._scip)

    def getReadingTime(self):
        """Retrieve the current reading time in seconds"""
        return SCIPgetReadingTime(self._scip)

    def getPresolvingTime(self):
        """Retrieve the curernt presolving time in seconds"""
        return SCIPgetPresolvingTime(self._scip)

    def getNNodes(self):
        """Retrieve the total number of processed nodes."""
        return SCIPgetNNodes(self._scip)

    def getUpperbound(self):
        """Retrieve the upper bound of the transformed problem."""
        return SCIPgetUpperbound(self._scip)

    def getLowerbound(self):
        """Retrieve the lower bound of the transformed problem."""
        return SCIPgetLowerbound(self._scip)

    def getCurrentNode(self):
        """Retrieve current node."""
        return Node.create(SCIPgetCurrentNode(self._scip))

    def getNLPIterations(self):
        """Retrieve current number of LP iterations."""
        return SCIPgetNLPIterations(self._scip)

    def getGap(self):
        """Retrieve the gap, i.e. |(primalbound - dualbound)/min(|primalbound|,|dualbound|)|."""
        return SCIPgetGap(self._scip)

    def getDepth(self):
        """Retrieve the depth of the current node"""
        return SCIPgetDepth(self._scip)

    def infinity(self):
        """Retrieve SCIP's infinity value"""
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """Retrieve epsilon for e.g. equality checks"""
        return SCIPepsilon(self._scip)

    def feastol(self):
        """Retrieve feasibility tolerance"""
        return SCIPfeastol(self._scip)

    def feasFrac(self, value):
        """returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol)"""
        return SCIPfeasFrac(self._scip, value)

    def frac(self, value):
        """returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps)"""
        return SCIPfrac(self._scip, value)

    def isZero(self, value):
        """returns whether abs(value) < eps"""
        return SCIPisZero(self._scip, value)

    def isFeasZero(self, value):
        """returns whether abs(value) < feastol"""
        return SCIPisFeasZero(self._scip, value)

    def isInfinity(self, value):
        """returns whether value is SCIP's infinity"""
        return SCIPisInfinity(self._scip, value)

    def isFeasNegative(self, value):
        """returns whether value < -feastol"""
        return SCIPisFeasNegative(self._scip, value)

    def isFeasIntegral(self, value):
        """returns whether value is integral"""
        return SCIPisFeasIntegral(self._scip, value)

    def isLE(self, val1, val2):
        """returns whether val1 <= val2 + eps"""
        return SCIPisLE(self._scip, val1, val2)

    def isLT(self, val1, val2):
        """returns whether val1 < val2 - eps"""
        return SCIPisLT(self._scip, val1, val2)

    def isGE(self, val1, val2):
        """returns whether val1 >= val2 - eps"""
        return SCIPisGE(self._scip, val1, val2)

    def isGT(self, val1, val2):
        """returns whether val1 > val2 + eps"""
        return SCIPisGT(self._scip, val1, val2)

    def getCondition(self, exact=False):
        """Get the current LP's condition number

        :param exact: whether to get an estimate or the exact value (Default value = False)

        """
        cdef SCIP_LPI* lpi
        PY_SCIP_CALL(SCIPgetLPI(self._scip, &lpi))
        cdef SCIP_Real quality = 0
        if exact:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_EXACTCONDITION, &quality))
        else:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_ESTIMCONDITION, &quality))

        return quality

    # Objective function

    def setMinimize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjlimit(self, objlimit):
        """Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        :param objlimit: limit on the objective function

        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def getObjlimit(self):
        """returns current limit on objective function."""
        return SCIPgetObjlimit(self._scip)

    def setObjective(self, coeffs, sense = 'minimize', clear = 'true'):
        """Establish the objective function as a linear expression.

        :param coeffs: the coefficients
        :param sense: the objective sense (Default value = 'minimize')
        :param clear: set all other variables objective coefficient to zero (Default value = 'true')

        """
        cdef SCIP_VAR** _vars
        cdef int _nvars

        # turn the constant value into an Expr instance for further processing
        if not isinstance(coeffs, Expr):
            assert(_is_number(coeffs)), "given coefficients are neither Expr or number but %s" % coeffs.__class__.__name__
            coeffs = Expr() + coeffs

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            self.addObjoffset(coeffs[CONST])

        if clear:
            # clear existing objective function
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)
            for i in range(_nvars):
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, _vars[i], 0.0))

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, var.scip_var, coef))

        if sense == "minimize":
            self.setMinimize()
        elif sense == "maximize":
            self.setMaximize()
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

    def getObjective(self):
        """Retrieve objective function as Expr"""
        variables = self.getVars()
        objective = Expr()
        for var in variables:
            coeff = var.getObj()
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        return objective

    def addObjoffset(self, offset, solutions = False):
        """Add constant offset to objective

        :param offset: offset to add
        :param solutions: add offset also to existing solutions (Default value = False)

        """
        if solutions:
            PY_SCIP_CALL(SCIPaddObjoffset(self._scip, offset))
        else:
            PY_SCIP_CALL(SCIPaddOrigObjoffset(self._scip, offset))

    def getObjoffset(self, original = True):
        """Retrieve constant objective offset

        :param original: offset of original or transformed problem (Default value = True)

        """
        if original:
            return SCIPgetOrigObjoffset(self._scip)
        else:
            return SCIPgetTransObjoffset(self._scip)

    # Setting parameters
    def setPresolve(self, setting):
        """Set presolving parameter settings.

        :param setting: the parameter settings (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetPresolving(self._scip, setting, True))

    def setProbName(self, name):
        """Set problem name"""
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetProbName(self._scip, n))

    def setSeparating(self, setting):
        """Set separating parameter settings.

        :param setting: the parameter settings (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetSeparating(self._scip, setting, True))

    def setHeuristics(self, setting):
        """Set heuristics parameter settings.

        :param setting: the parameter setting (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetHeuristics(self._scip, setting, True))

    def disablePropagation(self, onlyroot=False):
        """Disables propagation in SCIP to avoid modifying the original problem during transformation.

        :param onlyroot: use propagation when root processing is finished (Default value = False)

        """
        self.setIntParam("propagating/maxroundsroot", 0)
        if not onlyroot:
            self.setIntParam("propagating/maxrounds", 0)

    def writeProblem(self, filename='model.cip', trans=False):
        """Write current model/problem to a file.

        :param filename: the name of the file to be used (Default value = 'model.cip')
        :param trans: indicates whether the transformed problem is written to file (Default value = False)

        """
        fn = str_conversion(filename)
        fn, ext = splitext(fn)
        if len(ext) == 0:
            ext = str_conversion('.cip')
        fn = fn + ext
        ext = ext[1:]
        if trans:
            PY_SCIP_CALL(SCIPwriteTransProblem(self._scip, fn, ext, False))
        else:
            PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, fn, ext, False))
        print('wrote problem to file ' + str(fn))

    # Variable Functions

    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar = False):
        """Create a new variable. Default variable is non-negative and continuous.

        :param name: name of the variable, generic if empty (Default value = '')
        :param vtype: type of the variable (Default value = 'C')
        :param lb: lower bound of the variable, use None for -infinity (Default value = 0.0)
        :param ub: upper bound of the variable, use None for +infinity (Default value = None)
        :param obj: objective value of variable (Default value = 0.0)
        :param pricedVar: is the variable a pricing candidate? (Default value = False)

        """

        # replace empty name with generic one
        if name == '':
            name = 'x'+str(SCIPgetNVars(self._scip)+1)

        cname = str_conversion(name)
        if ub is None:
            ub = SCIPinfinity(self._scip)
        if lb is None:
            lb = -SCIPinfinity(self._scip)
        cdef SCIP_VAR* scip_var
        vtype = vtype.upper()
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS))
        elif vtype in ['B', 'BINARY']:
            lb = 0.0
            ub = 1.0
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_BINARY))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_INTEGER))
        else:
            raise Warning("unrecognized variable type")

        if pricedVar:
            PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))
        else:
            PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = Variable.create(scip_var)

        #setting the variable data
        SCIPvarSetData(scip_var, <SCIP_VARDATA*>pyVar)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def releaseVar(self, Variable var):
        """Release the variable.

        :param Variable var: variable to be released

        """
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &var.scip_var))

    def getTransformedVar(self, Variable var):
        """Retrieve the transformed variable.

        :param Variable var: original variable to get the transformed of

        """
        cdef SCIP_VAR* _tvar
        PY_SCIP_CALL(SCIPtransformVar(self._scip, var.scip_var, &_tvar))
        return Variable.create(_tvar)

    def addVarLocks(self, Variable var, nlocksdown, nlocksup):
        """adds given values to lock numbers of variable for rounding

        :param Variable var: variable to adjust the locks for
        :param nlocksdown: new number of down locks
        :param nlocksup: new number of up locks

        """
        PY_SCIP_CALL(SCIPaddVarLocks(self._scip, var.scip_var, nlocksdown, nlocksup))

    def fixVar(self, Variable var, val):
        """Fixes the variable var to the value val if possible.

        :param Variable var: variable to fix
        :param val: float, the fix value
        :return: tuple (infeasible, fixed) of booleans

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool fixed
        PY_SCIP_CALL(SCIPfixVar(self._scip, var.scip_var, val, &infeasible, &fixed))
        return infeasible, fixed

    def delVar(self, Variable var):
        """Delete a variable.

        :param var: the variable which shall be deleted
        :return: bool, was deleting succesful

        """
        cdef SCIP_Bool deleted
        PY_SCIP_CALL(SCIPdelVar(self._scip, var.scip_var, &deleted))
        return deleted

    def tightenVarLb(self, Variable var, lb, force=False):
        """Tighten the lower bound in preprocessing or current node, if the bound is tighter.

        :param var: SCIP variable
        :param lb: possible new lower bound
        :param force: force tightening even if below bound strengthening tolerance
        :return: tuple of bools, (infeasible, tightened)
                    infeasible: whether new domain is empty
                    tightened: whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLb(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened


    def tightenVarUb(self, Variable var, ub, force=False):
        """Tighten the upper bound in preprocessing or current node, if the bound is tighter.

        :param var: SCIP variable
        :param ub: possible new upper bound
        :param force: force tightening even if below bound strengthening tolerance
        :return: tuple of bools, (infeasible, tightened)
                    infeasible: whether new domain is empty
                    tightened: whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUb(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened


    def tightenVarUbGlobal(self, Variable var, ub, force=False):
        """Tighten the global upper bound, if the bound is tighter.

        :param var: SCIP variable
        :param ub: possible new upper bound
        :param force: force tightening even if below bound strengthening tolerance
        :return: tuple of bools, (infeasible, tightened)
                    infeasible: whether new domain is empty
                    tightened: whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUbGlobal(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarLbGlobal(self, Variable var, lb, force=False):
        """Tighten the global upper bound, if the bound is tighter.

        :param var: SCIP variable
        :param lb: possible new upper bound
        :param force: force tightening even if below bound strengthening tolerance
        :return: tuple of bools, (infeasible, tightened)
                    infeasible: whether new domain is empty
                    tightened: whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLbGlobal(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened

    def chgVarLb(self, Variable var, lb):
        """Changes the lower bound of the specified variable.

        :param Variable var: variable to change bound of
        :param lb: new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLb(self._scip, var.scip_var, lb))

    def chgVarUb(self, Variable var, ub):
        """Changes the upper bound of the specified variable.

        :param Variable var: variable to change bound of
        :param ub: new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUb(self._scip, var.scip_var, ub))


    def chgVarLbGlobal(self, Variable var, lb):
        """Changes the global lower bound of the specified variable.

        :param Variable var: variable to change bound of
        :param lb: new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbGlobal(self._scip, var.scip_var, lb))

    def chgVarUbGlobal(self, Variable var, ub):
        """Changes the global upper bound of the specified variable.

        :param Variable var: variable to change bound of
        :param ub: new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbGlobal(self._scip, var.scip_var, ub))

    def chgVarLbNode(self, Node node, Variable var, lb):
        """Changes the lower bound of the specified variable at the given node.

        :param Variable var: variable to change bound of
        :param lb: new lower bound (set to None for -infinity)
        """

        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbNode(self._scip, node.scip_node, var.scip_var, lb))

    def chgVarUbNode(self, Node node, Variable var, ub):
        """Changes the upper bound of the specified variable at the given node.

        :param Variable var: variable to change bound of
        :param ub: new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbNode(self._scip, node.scip_node, var.scip_var, ub))

    def chgVarType(self, Variable var, vtype):
        """Changes the type of a variable

        :param Variable var: variable to change type of
        :param vtype: new variable type

        """
        cdef SCIP_Bool infeasible
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_CONTINUOUS, &infeasible))
        elif vtype in ['B', 'BINARY']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_BINARY, &infeasible))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_INTEGER, &infeasible))
        else:
            raise Warning("unrecognized variable type")
        if infeasible:
            print('could not change variable type of variable %s' % var)

    def getVars(self, transformed=False):
        """Retrieve all variables.

        :param transformed: get transformed variables instead of original (Default value = False)

        """
        cdef SCIP_VAR** _vars
        cdef SCIP_VAR* _var
        cdef int _nvars
        vars = []

        if transformed:
            _vars = SCIPgetVars(self._scip)
            _nvars = SCIPgetNVars(self._scip)
        else:
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)

        return [Variable.create(_vars[i]) for i in range(_nvars)]

    def getNVars(self):
        """Retrieve number of variables in the problems"""
        return SCIPgetNVars(self._scip)

    def getNConss(self):
        """Retrieve the number of constraints."""
        return SCIPgetNConss(self._scip)

    def updateNodeLowerbound(self, Node node, lb):
        """if given value is larger than the node's lower bound (in transformed problem),
        sets the node's lower bound to the new value

        :param node: Node, the node to update
        :param newbound: float, new bound (if greater) for the node

        """
        PY_SCIP_CALL(SCIPupdateNodeLowerbound(self._scip, node.scip_node, lb))

    # LP Methods
    def getLPSolstat(self):
        """Gets solution status of current LP"""
        return SCIPgetLPSolstat(self._scip)


    def constructLP(self):
        """makes sure that the LP of the current node is loaded and
         may be accessed through the LP information methods

        :return:  bool cutoff, i.e. can the node be cut off?

        """
        cdef SCIP_Bool cutoff
        PY_SCIP_CALL(SCIPconstructLP(self._scip, &cutoff))
        return cutoff

    def getLPObjVal(self):
        """gets objective value of current LP (which is the sum of column and loose objective value)"""

        return SCIPgetLPObjval(self._scip)

    def getLPColsData(self):
        """Retrieve current LP columns"""
        cdef SCIP_COL** cols
        cdef int ncols

        PY_SCIP_CALL(SCIPgetLPColsData(self._scip, &cols, &ncols))
        return [Column.create(cols[i]) for i in range(ncols)]

    def getLPRowsData(self):
        """Retrieve current LP rows"""
        cdef SCIP_ROW** rows
        cdef int nrows

        PY_SCIP_CALL(SCIPgetLPRowsData(self._scip, &rows, &nrows))
        return [Row.create(rows[i]) for i in range(nrows)]

    def getNLPRows(self):
        """Retrieve the number of rows currently in the LP"""
        return SCIPgetNLPRows(self._scip)

    def getNLPCols(self):
        """Retrieve the number of cols currently in the LP"""
        return SCIPgetNLPCols(self._scip)

    def getLPBasisInd(self):
        """Gets all indices of basic columns and rows: index i >= 0 corresponds to column i, index i < 0 to row -i-1"""
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef int* inds = <int *> malloc(nrows * sizeof(int))

        PY_SCIP_CALL(SCIPgetLPBasisInd(self._scip, inds))
        result = [inds[i] for i in range(nrows)]
        free(inds)
        return result

    def getLPBInvRow(self, row):
        """gets a row from the inverse basis matrix B^-1"""
        # TODO: sparsity information
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPgetLPBInvRow(self._scip, row, coefs, NULL, NULL))
        result = [coefs[i] for i in range(nrows)]
        free(coefs)
        return result

    def getLPBInvARow(self, row):
        """gets a row from B^-1 * A"""
        # TODO: sparsity information
        cdef int ncols = SCIPgetNLPCols(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPgetLPBInvARow(self._scip, row, NULL, coefs, NULL, NULL))
        result = [coefs[i] for i in range(ncols)]
        free(coefs)
        return result

    def isLPSolBasic(self):
        """returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis"""
        return SCIPisLPSolBasic(self._scip)

    #TODO: documentation!!
    # LP Row Methods
    def createEmptyRowSepa(self, Sepa sepa, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """creates and captures an LP row without any coefficients from a separator

        :param sepa: separator that creates the row
        :param name: name of row (Default value = "row")
        :param lhs: left hand side of row (Default value = 0)
        :param rhs: right hand side of row (Default value = None)
        :param local: is row only valid locally? (Default value = True)
        :param modifiable: is row modifiable during node processing (subject to column generation)? (Default value = False)
        :param removable: should the row be removed from the LP due to aging or cleanup? (Default value = True)
        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        scip_sepa = SCIPfindSepa(self._scip, str_conversion(sepa.name))
        PY_SCIP_CALL(SCIPcreateEmptyRowSepa(self._scip, &row, scip_sepa, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def createEmptyRowUnspec(self, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """creates and captures an LP row without any coefficients from an unspecified source

        :param name: name of row (Default value = "row")
        :param lhs: left hand side of row (Default value = 0)
        :param rhs: right hand side of row (Default value = None)
        :param local: is row only valid locally? (Default value = True)
        :param modifiable: is row modifiable during node processing (subject to column generation)? (Default value = False)
        :param removable: should the row be removed from the LP due to aging or cleanup? (Default value = True)
        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        PY_SCIP_CALL(SCIPcreateEmptyRowUnspec(self._scip, &row, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def getRowActivity(self, Row row):
        """returns the activity of a row in the last LP or pseudo solution"""
        return SCIPgetRowActivity(self._scip, row.scip_row)

    def getRowLPActivity(self, Row row):
        """returns the activity of a row in the last LP solution"""
        return SCIPgetRowLPActivity(self._scip, row.scip_row)

    # TODO: do we need this? (also do we need release var??)
    def releaseRow(self, Row row not None):
        """decreases usage counter of LP row, and frees memory if necessary"""
        PY_SCIP_CALL(SCIPreleaseRow(self._scip, &row.scip_row))

    def cacheRowExtensions(self, Row row not None):
        """informs row, that all subsequent additions of variables to the row should be cached and not directly applied;
        after all additions were applied, flushRowExtensions() must be called;
        while the caching of row extensions is activated, information methods of the row give invalid results;
        caching should be used, if a row is build with addVarToRow() calls variable by variable to increase the performance"""
        PY_SCIP_CALL(SCIPcacheRowExtensions(self._scip, row.scip_row))

    def flushRowExtensions(self, Row row not None):
        """flushes all cached row extensions after a call of cacheRowExtensions() and merges coefficients with equal columns into a single coefficient"""
        PY_SCIP_CALL(SCIPflushRowExtensions(self._scip, row.scip_row))

    def addVarToRow(self, Row row not None, Variable var not None, value):
        """resolves variable to columns and adds them with the coefficient to the row"""
        PY_SCIP_CALL(SCIPaddVarToRow(self._scip, row.scip_row, var.scip_var, value))

    def printRow(self, Row row not None):
        """Prints row."""
        PY_SCIP_CALL(SCIPprintRow(self._scip, row.scip_row, NULL))

    # Cutting Plane Methods
    def addPoolCut(self, Row row not None):
        """if not already existing, adds row to global cut pool"""
        PY_SCIP_CALL(SCIPaddPoolCut(self._scip, row.scip_row))

    def getCutEfficacy(self, Row cut not None, Solution sol = None):
        """returns efficacy of the cut with respect to the given primal solution or the current LP solution: e = -feasibility/norm"""
        return SCIPgetCutEfficacy(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def isCutEfficacious(self, Row cut not None, Solution sol = None):
        """ returns whether the cut's efficacy with respect to the given primal solution or the current LP solution is greater than the minimal cut efficacy"""
        return SCIPisCutEfficacious(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def addCut(self, Row cut not None, forcecut = False):
        """adds cut to separation storage and returns whether cut has been detected to be infeasible for local bounds"""
        cdef SCIP_Bool infeasible
        PY_SCIP_CALL(SCIPaddRow(self._scip, cut.scip_row, forcecut, &infeasible))
        return infeasible

    def getNCuts(self):
        """Retrieve total number of cuts in storage"""
        return SCIPgetNCuts(self._scip)

    def getNCutsApplied(self):
        """Retrieve number of currently applied cuts"""
        return SCIPgetNCutsApplied(self._scip)

    # Constraint functions
    def addCons(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """Add a linear or quadratic constraint.

        :param cons: list of coefficients
        :param name: the name of the constraint, generic name if empty (Default value = '')
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked during for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be  moved to a more global node? (Default value = False)

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        # replace empty name with generic one
        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode)
        kwargs['lhs'] = -SCIPinfinity(self._scip) if cons._lhs is None else cons._lhs
        kwargs['rhs'] =  SCIPinfinity(self._scip) if cons._rhs is None else cons._rhs

        deg = cons.expr.degree()
        if deg <= 1:
            return self._addLinCons(cons, **kwargs)
        elif deg <= 2:
            return self._addQuadCons(cons, **kwargs)
        elif deg == float('inf'): # general nonlinear
            return self._addGenNonlinearCons(cons, **kwargs)
        else:
            return self._addNonlinearCons(cons, **kwargs)

    def _addLinCons(self, ExprCons lincons, **kwargs):
        assert isinstance(lincons, ExprCons), "given constraint is not ExprCons but %s" % lincons.__class__.__name__

        assert lincons.expr.degree() <= 1, "given constraint is not linear, degree == %d" % lincons.expr.degree()
        terms = lincons.expr.terms

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']), 0, NULL, NULL,
            kwargs['lhs'], kwargs['rhs'], kwargs['initial'],
            kwargs['separate'], kwargs['enforce'], kwargs['check'],
            kwargs['propagate'], kwargs['local'], kwargs['modifiable'],
            kwargs['dynamic'], kwargs['removable'], kwargs['stickingatnode']))

        for key, coeff in terms.items():
            var = <Variable>key[0]
            PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, scip_cons, var.scip_var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return PyCons

    def _addQuadCons(self, ExprCons quadcons, **kwargs):
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2, "given constraint is not quadratic, degree == %d" % quadcons.expr.degree()

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsQuadratic(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        for v, c in terms.items():
            if len(v) == 1: # linear
                var = <Variable>v[0]
                PY_SCIP_CALL(SCIPaddLinearVarQuadratic(self._scip, scip_cons, var.scip_var, c))
            else: # quadratic
                assert len(v) == 2, 'term length must be 1 or 2 but it is %s' % len(v)
                var1, var2 = <Variable>v[0], <Variable>v[1]
                PY_SCIP_CALL(SCIPaddBilinTermQuadratic(self._scip, scip_cons, var1.scip_var, var2.scip_var, c))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        return PyCons

    def _addNonlinearCons(self, ExprCons cons, **kwargs):
        cdef SCIP_EXPR* expr
        cdef SCIP_EXPR** varexprs
        cdef SCIP_EXPRDATA_MONOMIAL** monomials
        cdef int* idxs
        cdef SCIP_EXPRTREE* exprtree
        cdef SCIP_VAR** vars
        cdef SCIP_CONS* scip_cons

        terms = cons.expr.terms

        # collect variables
        variables = {var.ptr():var for term in terms for var in term}
        variables = list(variables.values())
        varindex = {var.ptr():idx for (idx,var) in enumerate(variables)}

        # create variable expressions
        varexprs = <SCIP_EXPR**> malloc(len(varindex) * sizeof(SCIP_EXPR*))
        for idx in varindex.values():
            PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &expr, SCIP_EXPR_VARIDX, <int>idx) )
            varexprs[idx] = expr

        # create monomials for terms
        monomials = <SCIP_EXPRDATA_MONOMIAL**> malloc(len(terms) * sizeof(SCIP_EXPRDATA_MONOMIAL*))
        for i, (term, coef) in enumerate(terms.items()):
            idxs = <int*> malloc(len(term) * sizeof(int))
            for j, var in enumerate(term):
                idxs[j] = varindex[var.ptr()]
            PY_SCIP_CALL( SCIPexprCreateMonomial(SCIPblkmem(self._scip), &monomials[i], <SCIP_Real>coef, <int>len(term), idxs, NULL) )
            free(idxs)

        # create polynomial from monomials
        PY_SCIP_CALL( SCIPexprCreatePolynomial(SCIPblkmem(self._scip), &expr,
                                               <int>len(varindex), varexprs,
                                               <int>len(terms), monomials, 0.0, <SCIP_Bool>True) )

        # create expression tree
        PY_SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(self._scip), &exprtree, expr, <int>len(variables), 0, NULL) )
        vars = <SCIP_VAR**> malloc(len(variables) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(variables): # same as varindex
            vars[idx] = (<Variable>var).scip_var
        PY_SCIP_CALL( SCIPexprtreeSetVars(exprtree, <int>len(variables), vars) )

        # create nonlinear constraint for exprtree
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL, # linear
            1, &exprtree, NULL, # nonlinear
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable'],
            kwargs['stickingatnode']) )
        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        PY_SCIP_CALL( SCIPexprtreeFree(&exprtree) )
        free(vars)
        free(monomials)
        free(varexprs)
        return PyCons

    def _addGenNonlinearCons(self, ExprCons cons, **kwargs):
        cdef SCIP_EXPR** childrenexpr
        cdef SCIP_EXPR** scipexprs
        cdef SCIP_EXPRTREE* exprtree
        cdef SCIP_CONS* scip_cons
        cdef int nchildren

        # get arrays from python's expression tree
        expr = cons.expr
        nodes = expr_to_nodes(expr)
        op2idx = Operator.operatorIndexDic

        # in nodes we have a list of tuples: each tuple is of the form
        # (operator, [indices]) where indices are the indices of the tuples
        # that are the children of this operator. This is sorted,
        # so we are going to do is:
        # loop over the nodes and create the expression of each
        # Note1: when the operator is SCIP_EXPR_CONST, [indices] stores the value
        # Note2: we need to compute the number of variable operators to find out
        # how many variables are there.
        nvars = 0
        for node in nodes:
            if op2idx[node[0]] == SCIP_EXPR_VARIDX:
                nvars += 1
        vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))

        varpos = 0
        scipexprs = <SCIP_EXPR**> malloc(len(nodes) * sizeof(SCIP_EXPR*))
        for i,node in enumerate(nodes):
            op = node[0]
            opidx = op2idx[op]
            if opidx == SCIP_EXPR_VARIDX:
                assert len(node[1]) == 1
                pyvar = node[1][0] # for vars we store the actual var!
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, <int>varpos) )
                vars[varpos] = (<Variable>pyvar).scip_var
                varpos += 1
                continue
            if opidx == SCIP_EXPR_CONST:
                assert len(node[1]) == 1
                value = node[1][0]
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, <SCIP_Real>value) )
                continue
            if opidx == SCIP_EXPR_SUM or opidx == SCIP_EXPR_PRODUCT:
                nchildren = len(node[1])
                childrenexpr = <SCIP_EXPR**> malloc(nchildren * sizeof(SCIP_EXPR*))
                for c, pos in enumerate(node[1]):
                    childrenexpr[c] = scipexprs[pos]
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, nchildren, childrenexpr) )

                free(childrenexpr);
                continue
            if opidx == SCIP_EXPR_REALPOWER:
                # the second child is the exponent which is a const
                valuenode = nodes[node[1][1]]
                assert op2idx[valuenode[0]] == SCIP_EXPR_CONST
                exponent = valuenode[1][0]
                if float(exponent).is_integer():
                    PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], SCIP_EXPR_INTPOWER, scipexprs[node[1][0]], <int>exponent) )
                else:
                    PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, scipexprs[node[1][0]], <SCIP_Real>exponent) )
                continue
            if opidx == SCIP_EXPR_EXP or opidx == SCIP_EXPR_LOG or opidx == SCIP_EXPR_SQRT or opidx == SCIP_EXPR_ABS:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, scipexprs[node[1][0]]) )
                continue
            # default:
            raise NotImplementedError
        assert varpos == nvars

        # create expression tree
        PY_SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(self._scip), &exprtree, scipexprs[len(nodes) - 1], nvars, 0, NULL) );
        PY_SCIP_CALL( SCIPexprtreeSetVars(exprtree, <int>nvars, vars) );

        # create nonlinear constraint for exprtree
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL, # linear
            1, &exprtree, NULL, # nonlinear
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable'],
            kwargs['stickingatnode']) )
        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        PY_SCIP_CALL( SCIPexprtreeFree(&exprtree) )

        # free more memory
        free(scipexprs)
        free(vars)

        return PyCons

    def addConsCoeff(self, Constraint cons, Variable var, coeff):
        """Add coefficient to the linear constraint (if non-zero).

        :param Constraint cons: constraint to be changed
        :param Variable var: variable to be added
        :param coeff: coefficient of new variable

        """
        PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons.scip_cons, var.scip_var, coeff))

    def addConsNode(self, Node node, Constraint cons, Node validnode=None):
        """Add a constraint to the given node

        :param Node node: node to add the constraint to
        :param Constraint cons: constraint to add
        :param Node validnode: more global node where cons is also valid

        """
        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, cons.scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, cons.scip_cons, NULL))

    def addConsLocal(self, Constraint cons, Node validnode=None):
        """Add a constraint to the current node

        :param Constraint cons: constraint to add
        :param Node validnode: more global node where cons is also valid

        """
        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, cons.scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, cons.scip_cons, NULL))

    def addConsSOS1(self, vars, weights=None, name="SOS1cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS1 constraint.

        :param vars: list of variables to be included
        :param weights: list of weights (Default value = None)
        :param name: name of the constraint (Default value = "SOS1cons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, scip_cons, var.scip_var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, scip_cons, var.scip_var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

    def addConsSOS2(self, vars, weights=None, name="SOS2cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS2 constraint.

        :param vars: list of variables to be included
        :param weights: list of weights (Default value = None)
        :param name: name of the constraint (Default value = "SOS2cons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: is the constraint only valid locally? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, scip_cons, var.scip_var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, scip_cons, var.scip_var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

    def addConsAnd(self, vars, resvar, name="ANDcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add an AND-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param resvar: BINARY variable (resultant)
        :param name: name of the constraint (Default value = "ANDcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).scip_var
        _resVar = (<Variable>resvar).scip_var

        PY_SCIP_CALL(SCIPcreateConsAnd(self._scip, &scip_cons, str_conversion(name), _resVar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsOr(self, vars, resvar, name="ORcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add an OR-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param resvar: BINARY variable (resultant)
        :param name: name of the constraint (Default value = "ORcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).scip_var
        _resVar = (<Variable>resvar).scip_var

        PY_SCIP_CALL(SCIPcreateConsOr(self._scip, &scip_cons, str_conversion(name), _resVar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsXor(self, vars, rhsvar, name="XORcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add a XOR-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param rhsvar: BOOLEAN value, explicit True, False or bool(obj) is needed (right-hand side)
        :param name: name of the constraint (Default value = "XORcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        assert type(rhsvar) is type(bool()), "Provide BOOLEAN value as rhsvar, you gave %s." % type(rhsvar)
        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).scip_var

        PY_SCIP_CALL(SCIPcreateConsXor(self._scip, &scip_cons, str_conversion(name), rhsvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsCardinality(self, consvars, cardval, indvars=None, weights=None, name="CardinalityCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        :param consvars: list of variables to be included
        :param cardval: nonnegative integer
        :param indvars: indicator variables indicating which variables may be treated as nonzero in cardinality constraint, or None if new indicator variables should be introduced automatically (Default value = None)
        :param weights: weights determining the variable order, or None if variables should be ordered in the same way they were added to the constraint (Default value = None)
        :param name: name of the constraint (Default value = "CardinalityCons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef SCIP_VAR* indvar

        PY_SCIP_CALL(SCIPcreateConsCardinality(self._scip, &scip_cons, str_conversion(name), 0, NULL, cardval, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        # circumvent an annoying bug in SCIP 4.0.0 that does not allow uninitialized weights
        if weights is None:
            weights = list(range(1, len(consvars) + 1))

        for i, v in enumerate(consvars):
            var = <Variable>v
            if indvars:
                indvar = (<Variable>indvars[i]).scip_var
            else:
                indvar = NULL
            if weights is None:
                PY_SCIP_CALL(SCIPappendVarCardinality(self._scip, scip_cons, var.scip_var, indvar))
            else:
                PY_SCIP_CALL(SCIPaddVarCardinality(self._scip, scip_cons, var.scip_var, indvar, <SCIP_Real>weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons


    def addConsIndicator(self, cons, binvar=None, name="IndicatorCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an indicator constraint for the linear inequality 'cons'.

        The 'binvar' argument models the redundancy of the linear constraint. A solution for which
        'binvar' is 1 must satisfy the constraint.

        :param cons: a linear inequality of the form "<="
        :param binvar: binary indicator variable, or None if it should be created (Default value = None)
        :param name: name of the constraint (Default value = "IndicatorCons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__
        cdef SCIP_CONS* scip_cons
        cdef SCIP_VAR* _binVar
        if cons._lhs is not None and cons._rhs is not None:
            raise ValueError("expected inequality that has either only a left or right hand side")

        if cons.expr.degree() > 1:
            raise ValueError("expected linear inequality, expression has degree %d" % cons.expr.degree())


        if cons._rhs is not None:
            rhs =  cons._rhs
            negate = False
        else:
            rhs = -cons._lhs
            negate = True

        _binVar = (<Variable>binvar).scip_var if binvar is not None else NULL

        PY_SCIP_CALL(SCIPcreateConsIndicator(self._scip, &scip_cons, str_conversion(name), _binVar, 0, NULL, NULL, rhs,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))
        terms = cons.expr.terms

        for key, coeff in terms.items():
            var = <Variable>key[0]
            if negate:
                coeff = -coeff
            PY_SCIP_CALL(SCIPaddVarIndicator(self._scip, scip_cons, var.scip_var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addPyCons(self, Constraint cons):
        """Adds a customly created cons.

        :param Constraint cons: constraint to add

        """
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons.scip_cons))
        Py_INCREF(cons)

    def addVarSOS1(self, Constraint cons, Variable var, weight):
        """Add variable to SOS1 constraint.

        :param Constraint cons: SOS1 constraint
        :param Variable var: new variable
        :param weight: weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS1(self, Constraint cons, Variable var):
        """Append variable to SOS1 constraint.

        :param Constraint cons: SOS1 constraint
        :param Variable var: variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons.scip_cons, var.scip_var))

    def addVarSOS2(self, Constraint cons, Variable var, weight):
        """Add variable to SOS2 constraint.

        :param Constraint cons: SOS2 constraint
        :param Variable var: new variable
        :param weight: weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS2(self, Constraint cons, Variable var):
        """Append variable to SOS2 constraint.

        :param Constraint cons: SOS2 constraint
        :param Variable var: variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons.scip_cons, var.scip_var))

    def setInitial(self, Constraint cons, newInit):
        """Set "initial" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newInit -- new initial value
        """
        PY_SCIP_CALL(SCIPsetConsInitial(self._scip, cons.scip_cons, newInit))

    def setRemovable(self, Constraint cons, newRem):
        """Set "removable" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newRem -- new removable value
        """
        PY_SCIP_CALL(SCIPsetConsRemovable(self._scip, cons.scip_cons, newRem))

    def setEnforced(self, Constraint cons, newEnf):
        """Set "enforced" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newEnf -- new enforced value
        """
        PY_SCIP_CALL(SCIPsetConsEnforced(self._scip, cons.scip_cons, newEnf))

    def setCheck(self, Constraint cons, newCheck):
        """Set "check" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newCheck -- new check value
        """
        PY_SCIP_CALL(SCIPsetConsChecked(self._scip, cons.scip_cons, newCheck))

    def chgRhs(self, Constraint cons, rhs):
        """Change right hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint
        :param rhs: new ride hand side (set to None for +infinity)

        """

        if rhs is None:
           rhs = SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgRhsLinear(self._scip, cons.scip_cons, rhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgRhsQuadratic(self._scip, cons.scip_cons, rhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgLhs(self, Constraint cons, lhs):
        """Change left hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint
        :param lhs: new left hand side (set to None for -infinity)

        """

        if lhs is None:
           lhs = -SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgLhsLinear(self._scip, cons.scip_cons, lhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgLhsQuadratic(self._scip, cons.scip_cons, lhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getRhs(self, Constraint cons):
        """Retrieve right hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetRhsLinear(self._scip, cons.scip_cons)
        elif constype == 'quadratic':
            return SCIPgetRhsQuadratic(self._scip, cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getLhs(self, Constraint cons):
        """Retrieve left hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetLhsLinear(self._scip, cons.scip_cons)
        elif constype == 'quadratic':
            return SCIPgetLhsQuadratic(self._scip, cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getActivity(self, Constraint cons, Solution sol = None):
        """Retrieve activity of given constraint.
        Can only be called after solving is completed.

        :param Constraint cons: linear or quadratic constraint
        :param Solution sol: solution to compute activity of, None to use current node's solution (Default value = None)

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol

        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPgetActivityQuadratic(self._scip, cons.scip_cons, scip_sol, &activity))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        return activity


    def getSlack(self, Constraint cons, Solution sol = None, side = None):
        """Retrieve slack of given constraint.
        Can only be called after solving is completed.


        :param Constraint cons: linear or quadratic constraint
        :param Solution sol: solution to compute slack of, None to use current node's solution (Default value = None)
        :param side: whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol


        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(self._scip, cons.scip_cons)
            rhs = SCIPgetRhsLinear(self._scip, cons.scip_cons)
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        elif constype == 'quadratic':
            lhs = SCIPgetLhsQuadratic(self._scip, cons.scip_cons)
            rhs = SCIPgetRhsQuadratic(self._scip, cons.scip_cons)
            PY_SCIP_CALL(SCIPgetActivityQuadratic(self._scip, cons.scip_cons, scip_sol, &activity))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        lhsslack = activity - lhs
        rhsslack = rhs - activity

        if side == 'lhs':
            return lhsslack
        elif side == 'rhs':
            return rhsslack
        else:
            return min(lhsslack, rhsslack)

    def getTransformedCons(self, Constraint cons):
        """Retrieve transformed constraint.

        :param Constraint cons: constraint

        """
        cdef SCIP_CONS* transcons
        PY_SCIP_CALL(SCIPgetTransformedCons(self._scip, cons.scip_cons, &transcons))
        return Constraint.create(transcons)

    def getTermsQuadratic(self, Constraint cons):
        """Retrieve bilinear, quadratic, and linear terms of a quadratic constraint.

        :param Constraint cons: constraint

        """
        cdef SCIP_QUADVARTERM* _quadterms
        cdef SCIP_BILINTERM* _bilinterms
        cdef SCIP_VAR** _linvars
        cdef SCIP_Real* _lincoefs
        cdef int _nbilinterms
        cdef int _nquadterms
        cdef int _nlinvars

        assert cons.isQuadratic(), "constraint is not quadratic"

        bilinterms = []
        quadterms  = []
        linterms   = []

        # bilinear terms
        _bilinterms = SCIPgetBilinTermsQuadratic(self._scip, cons.scip_cons)
        _nbilinterms = SCIPgetNBilinTermsQuadratic(self._scip, cons.scip_cons)

        for i in range(_nbilinterms):
            var1 = Variable.create(_bilinterms[i].var1)
            var2 = Variable.create(_bilinterms[i].var2)
            bilinterms.append((var1,var2,_bilinterms[i].coef))

        # quadratic terms
        _quadterms = SCIPgetQuadVarTermsQuadratic(self._scip, cons.scip_cons)
        _nquadterms = SCIPgetNQuadVarTermsQuadratic(self._scip, cons.scip_cons)

        for i in range(_nquadterms):
            var = Variable.create(_quadterms[i].var)
            quadterms.append((var,_quadterms[i].sqrcoef,_quadterms[i].lincoef))

        # linear terms
        _linvars = SCIPgetLinearVarsQuadratic(self._scip, cons.scip_cons)
        _lincoefs = SCIPgetCoefsLinearVarsQuadratic(self._scip, cons.scip_cons)
        _nlinvars = SCIPgetNLinearVarsQuadratic(self._scip, cons.scip_cons)

        for i in range(_nlinvars):
            var = Variable.create(_linvars[i])
            linterms.append((var,_lincoefs[i]))

        return (bilinterms, quadterms, linterms)

    def setRelaxSolVal(self, Variable var, val):
        """sets the value of the given variable in the global relaxation solution"""
        PY_SCIP_CALL(SCIPsetRelaxSolVal(self._scip, var.scip_var, val))

    def getConss(self):
        """Retrieve all constraints."""
        cdef SCIP_CONS** _conss
        cdef int _nconss
        conss = []

        _conss = SCIPgetConss(self._scip)
        _nconss = SCIPgetNConss(self._scip)
        return [Constraint.create(_conss[i]) for i in range(_nconss)]

    def getNConss(self):
        """Retrieve number of all constraints"""
        return SCIPgetNConss(self._scip)

    def delCons(self, Constraint cons):
        """Delete constraint from the model

        :param Constraint cons: constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelCons(self._scip, cons.scip_cons))

    def delConsLocal(self, Constraint cons):
        """Delete constraint from the current node and it's children

        :param Constraint cons: constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelConsLocal(self._scip, cons.scip_cons))

    def getValsLinear(self, Constraint cons):
        """Retrieve the coefficients of a linear constraint

        :param Constraint cons: linear constraint to get the coefficients of

        """
        cdef SCIP_Real* _vals
        cdef SCIP_VAR** _vars

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        _vals = SCIPgetValsLinear(self._scip, cons.scip_cons)
        _vars = SCIPgetVarsLinear(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(SCIPgetNVarsLinear(self._scip, cons.scip_cons)):
            valsdict[bytes(SCIPvarGetName(_vars[i])).decode('utf-8')] = _vals[i]
        return valsdict

    def getDualMultiplier(self, Constraint cons):
        """Retrieve the dual multiplier to a linear constraint.

        :param Constraint cons: linear constraint

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("dual solution values not available for constraints of type ", constype)
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
        else:
            transcons = cons
        return SCIPgetDualsolLinear(self._scip, transcons.scip_cons)

    def getDualsolLinear(self, Constraint cons):
        """Retrieve the dual solution to a linear constraint.

        :param Constraint cons: linear constraint

        """
        cdef SCIP_Real dualsolval
        cdef SCIP_Bool boundconstraint
        cdef int _nvars
        cdef SCIP_VAR** _vars
        cdef SCIP_Real* _vals
        cdef SCIP_Bool _success

        if self.version() > 6.0:
            PY_SCIP_CALL( SCIPgetDualSolVal(self._scip, cons.scip_cons, &dualsolval, &boundconstraint) )
            return dualsolval
        else:
            dual = 0.0

            constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
            if not constype == 'linear':
                raise Warning("dual solution values not available for constraints of type ", constype)

            try:
                _nvars = SCIPgetNVarsLinear(self._scip, cons.scip_cons)
                if cons.isOriginal():
                    transcons = <Constraint>self.getTransformedCons(cons)
                else:
                    transcons = cons
                dual = SCIPgetDualsolLinear(self._scip, transcons.scip_cons)
                if dual == 0.0 and _nvars == 1:
                    _vars = SCIPgetVarsLinear(self._scip, transcons.scip_cons)
                    _vals = SCIPgetValsLinear(self._scip, transcons.scip_cons)
                    activity = SCIPvarGetLPSol(_vars[0]) * _vals[0]
                    rhs = SCIPgetRhsLinear(self._scip, transcons.scip_cons)
                    lhs = SCIPgetLhsLinear(self._scip, transcons.scip_cons)
                    if (activity == rhs) or (activity == lhs):
                        dual = SCIPgetVarRedcost(self._scip, _vars[0])

                if self.getObjectiveSense() == "maximize" and not dual == 0.0:
                    dual = -dual
            except:
                raise Warning("no dual solution available for constraint " + cons.name)
            return dual

    def getDualfarkasLinear(self, Constraint cons):
        """Retrieve the dual farkas value to a linear constraint.

        :param Constraint cons: linear constraint

        """
        # TODO this should ideally be handled on the SCIP side
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
            return SCIPgetDualfarkasLinear(self._scip, transcons.scip_cons)
        else:
            return SCIPgetDualfarkasLinear(self._scip, cons.scip_cons)

    def getVarRedcost(self, Variable var):
        """Retrieve the reduced cost of a variable.

        :param Variable var: variable to get the reduced cost of

        """
        redcost = None
        try:
            redcost = SCIPgetVarRedcost(self._scip, var.scip_var)
            if self.getObjectiveSense() == "maximize":
                redcost = -redcost
        except:
            raise Warning("no reduced cost available for variable " + var.name)
        return redcost

    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(SCIPsolve(self._scip))
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))

    def presolve(self):
        """Presolve the problem."""
        PY_SCIP_CALL(SCIPpresolve(self._scip))

    # Benders' decomposition methods
    def initBendersDefault(self, subproblems):
        """initialises the default Benders' decomposition with a dictionary of subproblems

        Keyword arguments:
        subproblems -- a single Model instance or dictionary of Model instances
        """
        cdef SCIP** subprobs
        cdef SCIP_BENDERS* benders

        # checking whether subproblems is a dictionary
        if isinstance(subproblems, dict):
            isdict = True
            nsubproblems = len(subproblems)
        else:
            isdict = False
            nsubproblems = 1

        # create array of SCIP instances for the subproblems
        subprobs = <SCIP**> malloc(nsubproblems * sizeof(SCIP*))

        # if subproblems is a dictionary, then the dictionary is turned into a c array
        if isdict:
            for idx, subprob in enumerate(subproblems.values()):
                subprobs[idx] = (<Model>subprob)._scip
        else:
            subprobs[0] = (<Model>subproblems)._scip

        # creating the default Benders' decomposition
        PY_SCIP_CALL(SCIPcreateBendersDefault(self._scip, subprobs, nsubproblems))
        benders = SCIPfindBenders(self._scip, "default")

        # activating the Benders' decomposition constraint handlers
        self.setBoolParam("constraints/benderslp/active", True)
        self.setBoolParam("constraints/benders/active", True)
        #self.setIntParam("limits/maxorigsol", 0)

    def computeBestSolSubproblems(self):
        """Solves the subproblems with the best solution to the master problem.
        Afterwards, the best solution from each subproblem can be queried to get
        the solution to the original problem.

        If the user wants to resolve the subproblems, they must free them by
        calling freeBendersSubproblems()
        """
        cdef SCIP_BENDERS** _benders
        cdef SCIP_Bool _infeasible
        cdef int nbenders
        cdef int nsubproblems

        solvecip = True

        nbenders = SCIPgetNActiveBenders(self._scip)
        _benders = SCIPgetBenders(self._scip)

        # solving all subproblems from all Benders' decompositions
        for i in range(nbenders):
            nsubproblems = SCIPbendersGetNSubproblems(_benders[i])
            for j in range(nsubproblems):
                PY_SCIP_CALL(SCIPsetupBendersSubproblem(self._scip,
                    _benders[i], self._bestSol.sol, j))
                PY_SCIP_CALL(SCIPsolveBendersSubproblem(self._scip,
                    _benders[i], self._bestSol.sol, j, &_infeasible,
                    SCIP_BENDERSENFOTYPE_CHECK, solvecip, NULL))

    def freeBendersSubproblems(self):
        """Calls the free subproblem function for the Benders' decomposition.
        This will free all subproblems for all decompositions.
        """
        cdef SCIP_BENDERS** _benders
        cdef int nbenders
        cdef int nsubproblems

        nbenders = SCIPgetNActiveBenders(self._scip)
        _benders = SCIPgetBenders(self._scip)

        # solving all subproblems from all Benders' decompositions
        for i in range(nbenders):
            nsubproblems = SCIPbendersGetNSubproblems(_benders[i])
            for j in range(nsubproblems):
                PY_SCIP_CALL(SCIPfreeBendersSubproblem(self._scip, _benders[i],
                    j))

    def updateBendersLowerbounds(self, lowerbounds, Benders benders=None):
        """"updates the subproblem lower bounds for benders using
        the lowerbounds dict. If benders is None, then the default
        Benders' decomposition is updated
        """
        cdef SCIP_BENDERS* _benders

        assert type(lowerbounds) is dict

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            n = str_conversion(benders.name)
            _benders = SCIPfindBenders(self._scip, n)

        for d in lowerbounds.keys():
            SCIPbendersUpdateSubproblemLowerbound(_benders, d, lowerbounds[d])

    def activateBenders(self, str name, int nsubproblems):
        """Activates the Benders' decomposition plugin with the input name

        Keyword arguments:
        name -- the name of the Benders' decomposition plugin
        nsubproblems -- the number of subproblems in the Benders' decomposition
        """
        n = str_conversion(name)
        cdef SCIP_BENDERS* benders
        benders = SCIPfindBenders(self._scip, n)
        PY_SCIP_CALL(SCIPactivateBenders(self._scip, benders, nsubproblems))

    def addBendersSubproblem(self, str name, subproblem):
        """adds a subproblem to the Benders' decomposition given by the input
        name.

        Keyword arguments:
        name -- the Benders' decomposition that the subproblem is added to
        subproblem --  the subproblem to add to the decomposition
        """
        cdef SCIP_BENDERS* benders
        n = str_conversion(name)
        benders = SCIPfindBenders(self._scip, n)
        PY_SCIP_CALL(SCIPaddBendersSubproblem(self._scip, benders, (<Model>subproblem)._scip))

    def setupBendersSubproblem(self, probnumber, Benders benders = None, Solution solution = None):
        """ sets up the Benders' subproblem given the master problem solution

        Keyword arguments:
        probnumber -- the index of the problem that is to be set up
        benders -- the Benders' decomposition to which the subproblem belongs to
        solution -- the master problem solution that is used for the set up, if None, then the LP solution is used
        """
        cdef SCIP_BENDERS* scip_benders
        cdef SCIP_SOL* scip_sol

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        if benders is None:
            scip_benders = SCIPfindBenders(self._scip, "default")
        else:
            n = str_conversion(benders.name)
            scip_benders = SCIPfindBenders(self._scip, n)

        PY_SCIP_CALL(SCIPsetupBendersSubproblem(self._scip, scip_benders, scip_sol, probnumber))

    def solveBendersSubproblem(self, probnumber, enfotype, solvecip, Benders benders = None, Solution solution = None):
        """ solves the Benders' decomposition subproblem. The convex relaxation will be solved unless
        the parameter solvecip is set to True.

        Keyword arguments:
        probnumber -- the index of the problem that is to be set up
        enfotype -- the enforcement type used for solving the subproblem, see SCIP_BENDERSENFOTYPE
        solvecip -- should the CIP of the subproblem be solved, if False, then only the convex relaxation is solved
        benders -- the Benders' decomposition to which the subproblem belongs to
        solution -- the master problem solution that is used for the set up, if None, then the LP solution is used
        """

        cdef SCIP_BENDERS* scip_benders
        cdef SCIP_SOL* scip_sol
        cdef SCIP_Real objective
        cdef SCIP_Bool infeasible

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        if benders is None:
            scip_benders = SCIPfindBenders(self._scip, "default")
        else:
            n = str_conversion(benders.name)
            scip_benders = SCIPfindBenders(self._scip, n)

        PY_SCIP_CALL(SCIPsolveBendersSubproblem(self._scip, scip_benders, scip_sol,
            probnumber, &infeasible, enfotype, solvecip, &objective))

        return infeasible, objective

    def getBendersVar(self, Variable var, Benders benders = None, probnumber = -1):
        """Returns the variable for the subproblem or master problem
        depending on the input probnumber

        Keyword arguments:
        var -- the source variable for which the target variable is requested
        benders -- the Benders' decomposition to which the subproblem variables belong to
        probnumber -- the problem number for which the target variable belongs, -1 for master problem
        """
        cdef SCIP_BENDERS* _benders
        cdef SCIP_VAR* _mappedvar

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            n = str_conversion(benders.name)
            _benders = SCIPfindBenders(self._scip, n)

        if probnumber == -1:
            PY_SCIP_CALL(SCIPgetBendersMasterVar(self._scip, _benders, var.scip_var, &_mappedvar))
        else:
            PY_SCIP_CALL(SCIPgetBendersSubproblemVar(self._scip, _benders, var.scip_var, &_mappedvar, probnumber))

        if _mappedvar == NULL:
            mappedvar = None
        else:
            mappedvar = Variable.create(_mappedvar)

        return mappedvar


    def includeEventhdlr(self, Eventhdlr eventhdlr, name, desc):
        """Include an event handler.

        Keyword arguments:
        eventhdlr -- event handler
        name -- name of event handler
        desc -- description of event handler
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeEventhdlr(self._scip, n, d,
                                          PyEventCopy,
                                          PyEventFree,
                                          PyEventInit,
                                          PyEventExit,
                                          PyEventInitsol,
                                          PyEventExitsol,
                                          PyEventDelete,
                                          PyEventExec,
                                          <SCIP_EVENTHDLRDATA*>eventhdlr))
        eventhdlr.model = <Model>weakref.proxy(self)
        eventhdlr.name = name
        Py_INCREF(eventhdlr)

    def includePricer(self, Pricer pricer, name, desc, priority=1, delay=True):
        """Include a pricer.

        :param Pricer pricer: pricer
        :param name: name of pricer
        :param desc: description of pricer
        :param priority: priority of pricer (Default value = 1)
        :param delay: should the pricer be delayed until no other pricers or already existing problem variables with negative reduced costs are found? (Default value = True)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePricer(self._scip, n, d,
                                            priority, delay,
                                            PyPricerCopy, PyPricerFree, PyPricerInit, PyPricerExit, PyPricerInitsol, PyPricerExitsol, PyPricerRedcost, PyPricerFarkas,
                                            <SCIP_PRICERDATA*>pricer))
        cdef SCIP_PRICER* scip_pricer
        scip_pricer = SCIPfindPricer(self._scip, n)
        PY_SCIP_CALL(SCIPactivatePricer(self._scip, scip_pricer))
        pricer.model = <Model>weakref.proxy(self)
        Py_INCREF(pricer)

    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority=0,
                        enfopriority=0, chckpriority=0, sepafreq=-1, propfreq=-1,
                        eagerfreq=100, maxprerounds=-1, delaysepa=False,
                        delayprop=False, needscons=True,
                        proptiming=PY_SCIP_PROPTIMING.BEFORELP,
                        presoltiming=PY_SCIP_PRESOLTIMING.MEDIUM):
        """Include a constraint handler

        :param Conshdlr conshdlr: constraint handler
        :param name: name of constraint handler
        :param desc: description of constraint handler
        :param sepapriority: priority for separation (Default value = 0)
        :param enfopriority: priority for constraint enforcing (Default value = 0)
        :param chckpriority: priority for checking feasibility (Default value = 0)
        :param sepafreq: frequency for separating cuts; 0 = only at root node (Default value = -1)
        :param propfreq: frequency for propagating domains; 0 = only preprocessing propagation (Default value = -1)
        :param eagerfreq: frequency for using all instead of only the useful constraints in separation, propagation and enforcement; -1 = no eager evaluations, 0 = first only (Default value = 100)
        :param maxprerounds: maximal number of presolving rounds the constraint handler participates in (Default value = -1)
        :param delaysepa: should separation method be delayed, if other separators found cuts? (Default value = False)
        :param delayprop: should propagation method be delayed, if other propagators found reductions? (Default value = False)
        :param needscons: should the constraint handler be skipped, if no constraints are available? (Default value = True)
        :param proptiming: positions in the node solving loop where propagation method of constraint handlers should be executed (Default value = SCIP_PROPTIMING.BEFORELP)
        :param presoltiming: timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING.MEDIUM)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeConshdlr(self._scip, n, d, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                                              maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming,
                                              PyConshdlrCopy, PyConsFree, PyConsInit, PyConsExit, PyConsInitpre, PyConsExitpre,
                                              PyConsInitsol, PyConsExitsol, PyConsDelete, PyConsTrans, PyConsInitlp, PyConsSepalp, PyConsSepasol,
                                              PyConsEnfolp, PyConsEnforelax, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = <Model>weakref.proxy(self)
        conshdlr.name = name
        Py_INCREF(conshdlr)

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):
        """Create a constraint of a custom constraint handler

        :param Conshdlr conshdlr: constraint handler
        :param name: name of constraint
        :param initial:  (Default value = True)
        :param separate:  (Default value = True)
        :param enforce:  (Default value = True)
        :param check:  (Default value = True)
        :param propagate:  (Default value = True)
        :param local:  (Default value = False)
        :param modifiable:  (Default value = False)
        :param dynamic:  (Default value = False)
        :param removable:  (Default value = False)
        :param stickingatnode:  (Default value = False)

        """

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* scip_conshdlr
        scip_conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint()
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &(constraint.scip_cons), n, scip_conshdlr, <SCIP_CONSDATA*>constraint,
                                initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))
        return constraint

    def includePresol(self, Presol presol, name, desc, priority, maxrounds, timing=SCIP_PRESOLTIMING_FAST):
        """Include a presolver

        :param Presol presol: presolver
        :param name: name of presolver
        :param desc: description of presolver
        :param priority: priority of the presolver (>= 0: before, < 0: after constraint handlers)
        :param maxrounds: maximal number of presolving rounds the presolver participates in (-1: no limit)
        :param timing: timing mask of presolver (Default value = SCIP_PRESOLTIMING_FAST)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePresol(self._scip, n, d, priority, maxrounds, timing, PyPresolCopy, PyPresolFree, PyPresolInit,
                                            PyPresolExit, PyPresolInitpre, PyPresolExitpre, PyPresolExec, <SCIP_PRESOLDATA*>presol))
        presol.model = <Model>weakref.proxy(self)
        Py_INCREF(presol)

    def includeSepa(self, Sepa sepa, name, desc, priority=0, freq=10, maxbounddist=1.0, usessubscip=False, delay=False):
        """Include a separator

        :param Sepa sepa: separator
        :param name: name of separator
        :param desc: description of separator
        :param priority: priority of separator (>= 0: before, < 0: after constraint handlers)
        :param freq: frequency for calling separator
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separation
        :param usessubscip: does the separator use a secondary SCIP instance? (Default value = False)
        :param delay: should separator be delayed, if other separators found cuts? (Default value = False)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeSepa(self._scip, n, d, priority, freq, maxbounddist, usessubscip, delay, PySepaCopy, PySepaFree,
                                          PySepaInit, PySepaExit, PySepaInitsol, PySepaExitsol, PySepaExeclp, PySepaExecsol, <SCIP_SEPADATA*>sepa))
        sepa.model = <Model>weakref.proxy(self)
        sepa.name = name
        Py_INCREF(sepa)

    def includeProp(self, Prop prop, name, desc, presolpriority, presolmaxrounds,
                    proptiming, presoltiming=SCIP_PRESOLTIMING_FAST, priority=1, freq=1, delay=True):
        """Include a propagator.

        :param Prop prop: propagator
        :param name: name of propagator
        :param desc: description of propagator
        :param presolpriority: presolving priority of the propgator (>= 0: before, < 0: after constraint handlers)
        :param presolmaxrounds: maximal number of presolving rounds the propagator participates in (-1: no limit)
        :param proptiming: positions in the node solving loop where propagation method of constraint handlers should be executed
        :param presoltiming: timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING_FAST)
        :param priority: priority of the propagator (Default value = 1)
        :param freq: frequency for calling propagator (Default value = 1)
        :param delay: should propagator be delayed if other propagators have found reductions? (Default value = True)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeProp(self._scip, n, d,
                                          priority, freq, delay,
                                          proptiming, presolpriority, presolmaxrounds,
                                          presoltiming, PyPropCopy, PyPropFree, PyPropInit, PyPropExit,
                                          PyPropInitpre, PyPropExitpre, PyPropInitsol, PyPropExitsol,
                                          PyPropPresol, PyPropExec, PyPropResProp,
                                          <SCIP_PROPDATA*> prop))
        prop.model = <Model>weakref.proxy(self)
        Py_INCREF(prop)

    def includeHeur(self, Heur heur, name, desc, dispchar, priority=10000, freq=1, freqofs=0,
                    maxdepth=-1, timingmask=SCIP_HEURTIMING_BEFORENODE, usessubscip=False):
        """Include a primal heuristic.

        :param Heur heur: heuristic
        :param name: name of heuristic
        :param desc: description of heuristic
        :param dispchar: display character of heuristic
        :param priority: priority of the heuristic (Default value = 10000)
        :param freq: frequency for calling heuristic (Default value = 1)
        :param freqofs: frequency offset for calling heuristic (Default value = 0)
        :param maxdepth: maximal depth level to call heuristic at (Default value = -1)
        :param timingmask: positions in the node solving loop where heuristic should be executed (Default value = SCIP_HEURTIMING_BEFORENODE)
        :param usessubscip: does the heuristic use a secondary SCIP instance? (Default value = False)

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        dis = ord(str_conversion(dispchar))
        PY_SCIP_CALL(SCIPincludeHeur(self._scip, nam, des, dis,
                                          priority, freq, freqofs,
                                          maxdepth, timingmask, usessubscip,
                                          PyHeurCopy, PyHeurFree, PyHeurInit, PyHeurExit,
                                          PyHeurInitsol, PyHeurExitsol, PyHeurExec,
                                          <SCIP_HEURDATA*> heur))
        heur.model = <Model>weakref.proxy(self)
        heur.name = name
        Py_INCREF(heur)

    def includeRelax(self, Relax relax, name, desc, priority=10000, freq=1):
        """Include a relaxation handler.

        :param Relax relax: relaxation handler
        :param name: name of relaxation handler
        :param desc: description of relaxation handler
        :param priority: priority of the relaxation handler (negative: after LP, non-negative: before LP, Default value = 10000)
        :param freq: frequency for calling relaxation handler

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeRelax(self._scip, nam, des, priority, freq, PyRelaxCopy, PyRelaxFree, PyRelaxInit, PyRelaxExit,
                                          PyRelaxInitsol, PyRelaxExitsol, PyRelaxExec, <SCIP_RELAXDATA*> relax))
        relax.model = <Model>weakref.proxy(self)
        relax.name = name

        Py_INCREF(relax)

    def includeBranchrule(self, Branchrule branchrule, name, desc, priority, maxdepth, maxbounddist):
        """Include a branching rule.

        :param Branchrule branchrule: branching rule
        :param name: name of branching rule
        :param desc: description of branching rule
        :param priority: priority of branching rule
        :param maxdepth: maximal depth level up to which this branching rule should be used (or -1)
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBranchrule(self._scip, nam, des,
                                          priority, maxdepth, maxbounddist,
                                          PyBranchruleCopy, PyBranchruleFree, PyBranchruleInit, PyBranchruleExit,
                                          PyBranchruleInitsol, PyBranchruleExitsol, PyBranchruleExeclp, PyBranchruleExecext,
                                          PyBranchruleExecps, <SCIP_BRANCHRULEDATA*> branchrule))
        branchrule.model = <Model>weakref.proxy(self)
        Py_INCREF(branchrule)

    def getChildren(self):
        cdef SCIP_NODE** children
        cdef int nchildren
        PY_SCIP_CALL(SCIPgetChildren(self._scip, &children, &nchildren))
        return [Node.create(children[i]) for i in range(nchildren)]

    def includeBenders(self, Benders benders, name, desc, priority=1, cutlp=True, cutpseudo=True, cutrelax=True,
            shareaux=False):
        """Include a Benders' decomposition.

        Keyword arguments:
        benders -- the Benders decomposition
        name -- the name
        desc -- the description
        priority -- priority of the Benders' decomposition
        cutlp -- should Benders' cuts be generated from LP solutions
        cutpseudo -- should Benders' cuts be generated from pseudo solutions
        cutrelax -- should Benders' cuts be generated from relaxation solutions
        shareaux -- should the Benders' decomposition share the auxiliary variables of the highest priority Benders' decomposition
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBenders(self._scip, n, d,
                                            priority, cutlp, cutrelax, cutpseudo, shareaux,
                                            PyBendersCopy, PyBendersFree, PyBendersInit, PyBendersExit, PyBendersInitpre,
                                            PyBendersExitpre, PyBendersInitsol, PyBendersExitsol, PyBendersGetvar,
                                            PyBendersCreatesub, PyBendersPresubsolve, PyBendersSolvesubconvex,
                                            PyBendersSolvesub, PyBendersPostsolve, PyBendersFreesub,
                                            <SCIP_BENDERSDATA*>benders))
        cdef SCIP_BENDERS* scip_benders
        scip_benders = SCIPfindBenders(self._scip, n)
        benders.model = <Model>weakref.proxy(self)
        benders.name = name
        Py_INCREF(benders)

    def includeBenderscut(self, Benders benders, Benderscut benderscut, name, desc, priority=1, islpcut=True):
        """ Include a Benders' decomposition cutting method

        Keyword arguments:
        benders -- the Benders' decomposition that this cutting method is attached to
        benderscut --- the Benders' decomposition cutting method
        name -- the name
        desc -- the description
        priority -- priority of the Benders' decomposition
        islpcut -- is this cutting method suitable for generating cuts for convex relaxations?
        """
        cdef SCIP_BENDERS* _benders

        bendersname = str_conversion(benders.name)
        _benders = SCIPfindBenders(self._scip, bendersname)

        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBenderscut(self._scip, _benders, n, d, priority, islpcut,
                                            PyBenderscutCopy, PyBenderscutFree, PyBenderscutInit, PyBenderscutExit,
                                            PyBenderscutInitsol, PyBenderscutExitsol, PyBenderscutExec,
                                            <SCIP_BENDERSCUTDATA*>benderscut))

        cdef SCIP_BENDERSCUT* scip_benderscut
        scip_benderscut = SCIPfindBenderscut(_benders, n)
        benderscut.model = <Model>weakref.proxy(self)
        benderscut.benders = <Benders>weakref.proxy(benders)
        benderscut.name = name
        # TODO: It might be necessary in increment the reference to benders i.e Py_INCREF(benders)
        Py_INCREF(benderscut)


    def getLPBranchCands(self):
        """gets branching candidates for LP solution branching (fractional variables) along with solution values,
        fractionalities, and number of branching candidates; The number of branching candidates does NOT account
        for fractional implicit integer variables which should not be used for branching decisions. Fractional
        implicit integer variables are stored at the positions *nlpcands to *nlpcands + *nfracimplvars - 1
        branching rules should always select the branching candidate among the first npriolpcands of the candidate list

        :return tuple (lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars) where

            lpcands: list of variables of LP branching candidates
            lpcandssol: list of LP candidate solution values
            lpcandsfrac	list of LP candidate fractionalities
            nlpcands:    number of LP branching candidates
            npriolpcands: number of candidates with maximal priority
            nfracimplvars: number of fractional implicit integer variables

        """
        cdef int ncands
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars

        ncands = SCIPgetNLPBranchCands(self._scip)
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac

        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac,
                                          &nlpcands, &npriolpcands, &nfracimplvars))

        return ([Variable.create(lpcands[i]) for i in range(ncands)], [lpcandssol[i] for i in range(ncands)],
                [lpcandsfrac[i] for i in range(ncands)], nlpcands, npriolpcands, nfracimplvars)


    def getPseudoBranchCands(self):
        """gets branching candidates for pseudo solution branching (non-fixed variables) along with the number of candidates

        :return tuple (pseudocands, npseudocands, npriopseudocands) where

            pseudocands:      list of pseudo branching variable candidates
            npriopseudocands: number of candidates with maximal priority

        """
        cdef SCIP_VAR** cands
        cdef int ncands
        cdef int npriocands

        PY_SCIP_CALL(SCIPgetPseudoBranchCands(self._scip, &cands, &ncands, &npriocands))

        return ([Variable.create(cands[i]) for i in range(ncands)], npriocands)


    def branchVar(self, variable):
        """Branch on a non-continuous variable.

        :param variable: Variable to branch on
        :return: tuple(downchild, eqchild, upchild) of Nodes of the left, middle and right child.

        """
        cdef SCIP_NODE* downchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))
        cdef SCIP_NODE* eqchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))
        cdef SCIP_NODE* upchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))

        PY_SCIP_CALL(SCIPbranchVar(self._scip, (<Variable>variable).scip_var, &downchild, &eqchild, &upchild))
        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)


    def branchVarVal(self, variable, value):
        """Branches on variable using a value which separates the domain of the variable.

        :param variable: Variable to branch on
        :param value: float, value to branch on
        :return: tuple(downchild, eqchild, upchild) of Nodes of the left, middle and right child. Middle child only exists
                    if branch variable is integer

        """
        cdef SCIP_NODE* downchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))
        cdef SCIP_NODE* eqchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))
        cdef SCIP_NODE* upchild = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))

        PY_SCIP_CALL(SCIPbranchVarVal(self._scip, (<Variable>variable).scip_var, value, &downchild, &eqchild, &upchild))
        # TODO should the stuff be freed and how?
        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)

    def calcNodeselPriority(self, Variable variable, branchdir, targetvalue):
        """calculates the node selection priority for moving the given variable's LP value
        to the given target value;
        this node selection priority can be given to the SCIPcreateChild() call

        :param variable: variable on which the branching is applied
        :param branchdir: type of branching that was performed
        :param targetvalue: new value of the variable in the child node
        :return: node selection priority for moving the given variable's LP value to the given target value

        """
        return SCIPcalcNodeselPriority(self._scip, variable.scip_var, branchdir, targetvalue)

    def calcChildEstimate(self, Variable variable, targetvalue):
        """Calculates an estimate for the objective of the best feasible solution
        contained in the subtree after applying the given branching;
        this estimate can be given to the SCIPcreateChild() call

        :param variable: Variable to compute the estimate for
        :param targetvalue: new value of the variable in the child node
        :return: objective estimate of the best solution in the subtree after applying the given branching

        """
        return SCIPcalcChildEstimate(self._scip, variable.scip_var, targetvalue)

    def createChild(self, nodeselprio, estimate):
        """Create a child node of the focus node.

        :param nodeselprio: float, node selection priority of new node
        :param estimate: float, estimate for(transformed) objective value of best feasible solution in subtree
        :return: Node, the child which was created

        """
        cdef SCIP_NODE* child = <SCIP_NODE*> malloc(sizeof(SCIP_NODE))
        PY_SCIP_CALL(SCIPcreateChild(self._scip, &child, nodeselprio, estimate))
        return Node.create(child)

    # Diving methods (Diving is LP related)
    def startDive(self):
        """Initiates LP diving
        It allows the user to change the LP in several ways, solve, change again, etc, without affecting the actual LP that has. When endDive() is called,
        SCIP will undo all changes done and recover the LP it had before startDive
        """
        PY_SCIP_CALL(SCIPstartDive(self._scip))

    def endDive(self):
        """Quits probing and resets bounds and constraints to the focus node's environment"""
        PY_SCIP_CALL(SCIPendDive(self._scip))

    def chgVarObjDive(self, Variable var, newobj):
        """changes (column) variable's objective value in current dive"""
        PY_SCIP_CALL(SCIPchgVarObjDive(self._scip, var.scip_var, newobj))

    def chgVarLbDive(self, Variable var, newbound):
        """changes variable's current lb in current dive"""
        PY_SCIP_CALL(SCIPchgVarLbDive(self._scip, var.scip_var, newbound))

    def chgVarUbDive(self, Variable var, newbound):
        """changes variable's current ub in current dive"""
        PY_SCIP_CALL(SCIPchgVarUbDive(self._scip, var.scip_var, newbound))

    def getVarLbDive(self, Variable var):
        """returns variable's current lb in current dive"""
        return SCIPgetVarLbDive(self._scip, var.scip_var)

    def getVarUbDive(self, Variable var):
        """returns variable's current ub in current dive"""
        return SCIPgetVarUbDive(self._scip, var.scip_var)

    def chgRowLhsDive(self, Row row, newlhs):
        """changes row lhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs()
        """
        PY_SCIP_CALL(SCIPchgRowLhsDive(self._scip, row.scip_row, newlhs))

    def chgRowRhsDive(self, Row row, newrhs):
        """changes row rhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs()
        """
        PY_SCIP_CALL(SCIPchgRowRhsDive(self._scip, row.scip_row, newrhs))

    def addRowDive(self, Row row):
        """adds a row to the LP in current dive"""
        PY_SCIP_CALL(SCIPaddRowDive(self._scip, row.scip_row))

    def solveDiveLP(self, itlim = -1):
        """solves the LP of the current dive no separation or pricing is applied
        no separation or pricing is applied
        :param itlim: maximal number of LP iterations to perform (Default value = -1, that is, no limit)
        returns two booleans:
        lperror -- if an unresolved lp error occured
        cutoff -- whether the LP was infeasible or the objective limit was reached
        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL(SCIPsolveDiveLP(self._scip, itlim, &lperror, &cutoff))
        return lperror, cutoff

    def inRepropagation(self):
        """returns if the current node is already solved and only propagated again."""
        return SCIPinRepropagation(self._scip)

    # Probing methods (Probing is tree based)
    def startProbing(self):
        """Initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
           SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), SCIPsolveProbingLP(), etc available
        """
        PY_SCIP_CALL(SCIPstartProbing(self._scip))

    def endProbing(self):
        """Quits probing and resets bounds and constraints to the focus node's environment"""
        PY_SCIP_CALL(SCIPendProbing(self._scip))

    def chgVarObjProbing(self, Variable var, newobj):
        """changes (column) variable's objective value during probing mode"""
        PY_SCIP_CALL(SCIPchgVarObjProbing(self._scip, var.scip_var, newobj))

    def fixVarProbing(self, Variable var, fixedval):
        """Fixes a variable at the current probing node."""
        PY_SCIP_CALL(SCIPfixVarProbing(self._scip, var.scip_var, fixedval))

    def isObjChangedProbing(self):
        """returns whether the objective function has changed during probing mode"""
        return SCIPisObjChangedProbing(self._scip)

    def inProbing(self):
        """returns whether we are in probing mode; probing mode is activated via startProbing() and stopped via endProbing()"""
        return SCIPinProbing(self._scip)

    def solveProbingLP(self, itlim = -1):
        """solves the LP at the current probing node (cannot be applied at preprocessing stage)
        no separation or pricing is applied
        :param itlim: maximal number of LP iterations to perform (Default value = -1, that is, no limit)
        returns two booleans:
        lperror -- if an unresolved lp error occured
        cutoff -- whether the LP was infeasible or the objective limit was reached
        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL(SCIPsolveProbingLP(self._scip, itlim, &lperror, &cutoff))
        return lperror, cutoff

    def interruptSolve(self):
        """Interrupt the solving process as soon as possible."""
        PY_SCIP_CALL(SCIPinterruptSolve(self._scip))

    # Solution functions

    def createSol(self, Heur heur = None):
        """Create a new primal solution.

        :param Heur heur: heuristic that found the solution (Default value = None)

        """
        cdef SCIP_HEUR* _heur

        if isinstance(heur, Heur):
            n = str_conversion(heur.name)
            _heur = SCIPfindHeur(self._scip, n)
        else:
            _heur = NULL
        solution = Solution()
        PY_SCIP_CALL(SCIPcreateSol(self._scip, &solution.sol, _heur))
        return solution

    def printBestSol(self, write_zeros=False):
        """Prints the best feasible primal solution."""
        PY_SCIP_CALL(SCIPprintBestSol(self._scip, NULL, write_zeros))

    def printSol(self, Solution solution, write_zeros=False):
      """Print the given primal solution.

      Keyword arguments:
      solution -- solution to print
      write_zeros -- include variables that are set to zero
      """
      PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, NULL, write_zeros))

    def writeBestSol(self, filename="origprob.sol", write_zeros=False):
        """Write the best feasible primal solution to a file.

        Keyword arguments:
        filename -- name of the output file
        write_zeros -- include variables that are set to zero
        """
        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestSol(self._scip, cfile, write_zeros))

    def writeSol(self, Solution solution, filename="origprob.sol", write_zeros=False):
        """Write the given primal solution to a file.

        Keyword arguments:
        solution -- solution to write
        filename -- name of the output file
        write_zeros -- include variables that are set to zero
        """
        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, cfile, write_zeros))

    # perhaps this should not be included as it implements duplicated functionality
    #   (as does it's namesake in SCIP)
    def readSol(self, filename):
        """Reads a given solution file, problem has to be transformed in advance.

        Keyword arguments:
        filename -- name of the input file
        """
        fn = str_conversion(filename)
        PY_SCIP_CALL(SCIPreadSol(self._scip, fn))

    def readSolFile(self, filename):
        """Reads a given solution file.

        Solution is created but not added to storage/the model.
        Use 'addSol' OR 'trySol' to add it.

        Keyword arguments:
        filename -- name of the input file
        """
        cdef SCIP_Bool partial
        cdef SCIP_Bool error
        cdef SCIP_Bool stored
        cdef Solution solution

        fn = str_conversion(filename)
        solution = self.createSol()
        PY_SCIP_CALL(SCIPreadSolFile(self._scip, fn, solution.sol, False, &partial, &error))
        if error:
            raise Exception("SCIP: reading solution from file failed!")

        return solution

    def setSolVal(self, Solution solution, Variable var, val):
        """Set a variable in a solution.

        :param Solution solution: solution to be modified
        :param Variable var: variable in the solution
        :param val: value of the specified variable

        """
        cdef SCIP_SOL* _sol
        _sol = <SCIP_SOL*>solution.sol
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, var.scip_var, val))

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, free=True):
        """Check given primal solution for feasibility and try to add it to the storage.

        :param Solution solution: solution to store
        :param printreason: should all reasons of violations be printed? (Default value = True)
        :param completely: should all violation be checked? (Default value = False)
        :param checkbounds: should the bounds of the variables be checked? (Default value = True)
        :param checkintegrality: has integrality to be checked? (Default value = True)
        :param checklprows: have current LP rows (both local and global) to be checked? (Default value = True)
        :param free: should solution be freed? (Default value = True)

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        else:
            PY_SCIP_CALL(SCIPtrySol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        return stored

    def checkSol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, original=False):
        """Check given primal solution for feasibility without adding it to the storage.

        :param Solution solution: solution to store
        :param printreason: should all reasons of violations be printed? (Default value = True)
        :param completely: should all violation be checked? (Default value = False)
        :param checkbounds: should the bounds of the variables be checked? (Default value = True)
        :param checkintegrality: has integrality to be checked? (Default value = True)
        :param checklprows: have current LP rows (both local and global) to be checked? (Default value = True)
        :param original: must the solution be checked against the original problem (Default value = False)

        """
        cdef SCIP_Bool feasible
        if original:
            PY_SCIP_CALL(SCIPcheckSolOrig(self._scip, solution.sol, &feasible, printreason, completely))
        else:
            PY_SCIP_CALL(SCIPcheckSol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &feasible))
        return feasible

    def addSol(self, Solution solution, free=True):
        """Try to add a solution to the storage.

        :param Solution solution: solution to store
        :param free: should solution be freed afterwards? (Default value = True)

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPaddSolFree(self._scip, &solution.sol, &stored))
        else:
            PY_SCIP_CALL(SCIPaddSol(self._scip, solution.sol, &stored))
        return stored

    def freeSol(self, Solution solution):
        """Free given solution

        :param Solution solution: solution to be freed

        """
        PY_SCIP_CALL(SCIPfreeSol(self._scip, &solution.sol))

    def getSols(self):
        """Retrieve list of all feasible primal solutions stored in the solution storage."""
        cdef SCIP_SOL** _sols
        cdef SCIP_SOL* _sol
        _sols = SCIPgetSols(self._scip)
        nsols = SCIPgetNSols(self._scip)
        sols = []

        for i in range(nsols):
            sols.append(Solution.create(_sols[i]))

        return sols

    def getBestSol(self):
        """Retrieve currently best known feasible primal solution."""
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))
        return self._bestSol

    def getSolObjVal(self, Solution sol, original=True):
        """Retrieve the objective value of the solution.

        :param Solution sol: solution
        :param original: objective value in original space (Default value = True)

        """
        if sol == None:
            sol = Solution.create(NULL)
        if original:
            objval = SCIPgetSolOrigObj(self._scip, sol.sol)
        else:
            objval = SCIPgetSolTransObj(self._scip, sol.sol)
        return objval

    def getObjVal(self, original=True):
        """Retrieve the objective value of value of best solution.
        Can only be called after solving is completed.

        :param original: objective value in original space (Default value = True)

        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolObjVal(self._bestSol, original)

    def getSolVal(self, Solution sol, Variable var):
        """Retrieve value of given variable in the given solution or in
        the LP/pseudo solution if sol == None

        :param Solution sol: solution
        :param Variable var: variable to query the value of

        """
        if sol == None:
            sol = Solution.create(NULL)
        return SCIPgetSolVal(self._scip, sol.sol, var.scip_var)

    def getVal(self, Variable var):
        """Retrieve the value of the best known solution.
        Can only be called after solving is completed.

        :param Variable var: variable to query the value of

        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolVal(self._bestSol, var)

    def getPrimalbound(self):
        """Retrieve the best primal bound."""
        return SCIPgetPrimalbound(self._scip)

    def getDualbound(self):
        """Retrieve the best dual bound."""
        return SCIPgetDualbound(self._scip)

    def getDualboundRoot(self):
        """Retrieve the best root dual bound."""
        return SCIPgetDualboundRoot(self._scip)

    def writeName(self, Variable var):
        """Write the name of the variable to the std out.

        :param Variable var: variable

        """
        PY_SCIP_CALL(SCIPwriteVarName(self._scip, NULL, var.scip_var, False))

    def getStage(self):
        """Retrieve current SCIP stage"""
        return SCIPgetStage(self._scip)

    def getStatus(self):
        """Retrieve solution status."""
        cdef SCIP_STATUS stat = SCIPgetStatus(self._scip)
        if stat == SCIP_STATUS_OPTIMAL:
            return "optimal"
        elif stat == SCIP_STATUS_TIMELIMIT:
            return "timelimit"
        elif stat == SCIP_STATUS_INFEASIBLE:
            return "infeasible"
        elif stat == SCIP_STATUS_UNBOUNDED:
            return "unbounded"
        elif stat == SCIP_STATUS_USERINTERRUPT:
            return "userinterrupt"
        else:
            return "unknown"

    def getObjectiveSense(self):
        """Retrieve objective sense."""
        cdef SCIP_OBJSENSE sense = SCIPgetObjsense(self._scip)
        if sense == SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    def catchEvent(self, eventtype, Eventhdlr eventhdlr):
        """catches a global (not variable or row dependent) event"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchEvent(self._scip, eventtype, _eventhdlr, NULL, NULL))

    def dropEvent(self, eventtype, Eventhdlr eventhdlr):
        """drops a global event (stops to track event)"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropEvent(self._scip, eventtype, _eventhdlr, NULL, -1))

    def catchVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        """catches an objective value or domain change event on the given transformed variable"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchVarEvent(self._scip, var.scip_var, eventtype, _eventhdlr, NULL, NULL))

    def dropVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        """drops an objective value or domain change event (stops to track event) on the given transformed variable"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropVarEvent(self._scip, var.scip_var, eventtype, _eventhdlr, NULL, -1))

    def catchRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        """catches a row coefficient, constant, or side change event on the given row"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchRowEvent(self._scip, row.scip_row, eventtype, _eventhdlr, NULL, NULL))

    def dropRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        """drops a row coefficient, constant, or side change event (stops to track event) on the given row"""
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropRowEvent(self._scip, row.scip_row, eventtype, _eventhdlr, NULL, -1))

    # Statistic Methods

    def printStatistics(self):
        """Print statistics."""
        PY_SCIP_CALL(SCIPprintStatistics(self._scip, NULL))

    def writeStatistics(self, filename="origprob.stats"):
      """Write statistics to a file.

      Keyword arguments:
      filename -- name of the output file
      """
      # use this doubled opening pattern to ensure that IOErrors are
      #   triggered early and in Python not in C,Cython or SCIP.
      with open(filename, "w") as f:
          cfile = fdopen(f.fileno(), "w")
          PY_SCIP_CALL(SCIPprintStatistics(self._scip, cfile))

    def getNLPs(self):
        """gets total number of LPs solved so far"""
        return SCIPgetNLPs(self._scip)

    # Verbosity Methods

    def hideOutput(self, quiet = True):
        """Hide the output.

        :param quiet: hide output? (Default value = True)

        """
        SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Output Methods

    def redirectOutput(self):
        """Send output to python instead of terminal."""

        cdef SCIP_MESSAGEHDLR *myMessageHandler

        PY_SCIP_CALL(SCIPmessagehdlrCreate(&myMessageHandler, False, NULL, False, relayMessage, relayMessage, relayMessage, NULL, NULL))
        PY_SCIP_CALL(SCIPsetMessagehdlr(self._scip, myMessageHandler))
        SCIPmessageSetErrorPrinting(relayErrorMessage, NULL)

    # Parameter Methods

    def setBoolParam(self, name, value):
        """Set a boolean-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, value))

    def setIntParam(self, name, value):
        """Set an int-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, value))

    def setLongintParam(self, name, value):
        """Set a long-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, value))

    def setRealParam(self, name, value):
        """Set a real-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, value))

    def setCharParam(self, name, value):
        """Set a char-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, ord(value)))

    def setStringParam(self, name, value):
        """Set a string-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        v = str_conversion(value)
        PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, v))

    def setParam(self, name, value):
        """Set a parameter with value in int, bool, real, long, char or str.

        :param name: name of parameter
        :param value: value of parameter
        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, bool(int(value))))
        elif paramtype == SCIP_PARAMTYPE_INT:
            PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, int(value)))
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, int(value)))
        elif paramtype == SCIP_PARAMTYPE_REAL:
            PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, float(value)))
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_STRING:
            PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, value))


    def getParam(self, name):
        """Get the value of a parameter of type
        int, bool, real, long, char or str.

        :param name: name of parameter
        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            return SCIPparamGetBool(param)
        elif paramtype == SCIP_PARAMTYPE_INT:
            return SCIPparamGetInt(param)
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            return SCIPparamGetLongint(param)
        elif paramtype == SCIP_PARAMTYPE_REAL:
            return SCIPparamGetReal(param)
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            return chr(SCIPparamGetChar(param))
        elif paramtype == SCIP_PARAMTYPE_STRING:
            return SCIPparamGetString(param)


    def readParams(self, file):
        """Read an external parameter file.

        :param file: file to be read

        """
        absfile = str_conversion(abspath(file))
        PY_SCIP_CALL(SCIPreadParams(self._scip, absfile))

    def writeParams(self, filename='param.set', comments = True, onlychanged = True):
        """Write parameter settings to an external file.

        :param filename: file to be written (Default value = 'param.set')
        :param comments: write parameter descriptions as comments? (Default value = True)
        :param onlychanged: write only modified parameters (Default value = True)

        """
        fn = str_conversion(filename)
        PY_SCIP_CALL(SCIPwriteParams(self._scip, fn, comments, onlychanged))
        print('wrote parameter settings to file ' + filename)

    def resetParam(self, name):
        """Reset parameter setting to its default value

        :param name: parameter to reset

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPresetParam(self._scip, n))

    def resetParams(self):
        """Reset parameter settings to their default values"""
        PY_SCIP_CALL(SCIPresetParams(self._scip))

    def setEmphasis(self, paraemphasis, quiet = True):
        """Set emphasis settings

        :param paraemphasis: emphasis to set
        :param quiet: hide output? (Default value = True)

        """
        PY_SCIP_CALL(SCIPsetEmphasis(self._scip, paraemphasis, quiet))

    def readProblem(self, file, extension = None):
        """Read a problem instance from an external file.

        :param file: file to be read
        :param extension: specify file extension/type (Default value = None)

        """
        absfile = str_conversion(abspath(file))
        if extension is None:
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = str_conversion(extension)
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, extension))

    # Counting functions

    def count(self):
        """Counts the number of feasible points of problem."""
        PY_SCIP_CALL(SCIPcount(self._scip))

    def getNCountedSols(self):
        """Get number of feasible solution."""
        cdef SCIP_Bool valid
        cdef SCIP_Longint nsols

        nsols = SCIPgetNCountedSols(self._scip, &valid)
        if not valid:
            print('total number of solutions found is not valid!')
        return nsols

    def setParamsCountsols(self):
        """Sets SCIP parameters such that a valid counting process is possible."""
        PY_SCIP_CALL(SCIPsetParamsCountsols(self._scip))

    def freeReoptSolve(self):
        """Frees all solution process data and prepares for reoptimization"""
        PY_SCIP_CALL(SCIPfreeReoptSolve(self._scip))

    def chgReoptObjective(self, coeffs, sense = 'minimize'):
        """Establish the objective function as a linear expression.

        :param coeffs: the coefficients
        :param sense: the objective sense (Default value = 'minimize')

        """

        cdef SCIP_OBJSENSE objsense

        if sense == "minimize":
            objsense = SCIP_OBJSENSE_MINIMIZE
        elif sense == "maximize":
            objsense = SCIP_OBJSENSE_MAXIMIZE
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

        assert isinstance(coeffs, Expr), "given coefficients are not Expr but %s" % coeffs.__class__.__name__

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            raise ValueError("Constant offsets in objective are not supported!")

        cdef SCIP_VAR** _vars
        cdef int _nvars
        _vars = SCIPgetOrigVars(self._scip)
        _nvars = SCIPgetNOrigVars(self._scip)
        _coeffs = <SCIP_Real*> malloc(_nvars * sizeof(SCIP_Real))

        for i in range(_nvars):
            _coeffs[i] = 0.0

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                for i in range(_nvars):
                    if _vars[i] == var.scip_var:
                        _coeffs[i] = coef

        PY_SCIP_CALL(SCIPchgReoptObjective(self._scip, objsense, _vars, &_coeffs[0], _nvars))

        free(_coeffs)

    def getVariablePseudocost(self, variable):
        cdef float ps_up, ps_down
        cdef SCIP_VAR* scip_var = (<Variable>variable).scip_var

        ps_up = SCIPgetVarPseudocost(self._scip, scip_var, SCIP_BRANCHDIR_UPWARDS)
        ps_down = SCIPgetVarPseudocost(self._scip, scip_var, SCIP_BRANCHDIR_DOWNWARDS)
        return ps_up * ps_down


    def includeNodesel(self, Nodesel nodesel, name, desc, stdpriority, memsavepriority):
        """Include a node selector.
        :param Nodesel nodesel: node selector
        :param name: name of node selector
        :param desc: description of node selector
        :param stdpriority: priority of the node selector in standard mode
        :param memsavepriority: priority of the node selector in memory saving mode
        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeNodesel(self._scip, nam, des,
                                          stdpriority, memsavepriority,
                                          PyNodeselCopy, PyNodeselFree, PyNodeselInit, PyNodeselExit,
                                          PyNodeselInitsol, PyNodeselExitsol, PyNodeselSelect, PyNodeselComp,
                                          <SCIP_NODESELDATA*> nodesel))
        nodesel.model = <Model>weakref.proxy(self)
        Py_INCREF(nodesel)

    def getMapping(self):
        cdef SCIP* scip = self._scip
        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)
        cdef int i, col_i
        cdef SCIP_VAR* scip_var

        cdef dict cands = {}

        for i in range(ncols):
            col_i = SCIPcolGetLPPos(cols[i])
            scip_var = SCIPcolGetVar(cols[i])
            var = Variable.create(scip_var)
            varname = var.name
            vartype = var.vtype()

            if vartype == 'BINARY':
                cands[col_i] = varname 

        return cands

    def getTimeSol(self):
        # check if a better sol is found
        cdef SCIP_SOL* sol = SCIPgetBestSol(self._scip)
        if sol != NULL:
            obj = SCIPgetSolOrigObj(self._scip, sol)
            time = SCIPgetSolTime(self._scip, sol)
            return (time, obj)
    
    def getNPrimalSols(self):
        return SCIPgetNSols(self._scip)

    def setup_ml_nodelsel(self, flag=0, indicator_arr=None):
       
        # find nodesel strategy and setup its priority
        cdef SCIP_NODESEL* nodesel = SCIPfindNodesel(self._scip, "nodeselml")
        assert(nodesel!=NULL)
        SCIPsetNodeselStdPriority(self._scip, nodesel, 6666666)
        # cdef SCIP_NODESELDATA* data = SCIPnodeselGetData(nodesel)
        # data.flag = flag


    def getState(self, prev_state = None):
        cdef SCIP* scip = self._scip
        cdef int i, j, k, col_i
        cdef SCIP_Real sim, prod

        update = prev_state is not None

        # COLUMNS
        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)

        cdef np.ndarray[np.int32_t,   ndim=1] col_types
        cdef np.ndarray[np.float32_t, ndim=1] col_coefs
        cdef np.ndarray[np.float32_t, ndim=1] col_lbs
        cdef np.ndarray[np.float32_t, ndim=1] col_ubs
        cdef np.ndarray[np.int32_t,   ndim=1] col_basestats
        cdef np.ndarray[np.float32_t, ndim=1] col_redcosts
        cdef np.ndarray[np.int32_t,   ndim=1] col_ages
        cdef np.ndarray[np.float32_t, ndim=1] col_solvals
        cdef np.ndarray[np.float32_t, ndim=1] col_solfracs
        cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_lb
        cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_ub
        cdef np.ndarray[np.float32_t, ndim=1] col_incvals
        cdef np.ndarray[np.float32_t, ndim=1] col_avgincvals

        if not update:
            col_types        = np.empty(shape=(ncols, ), dtype=np.int32)
            col_coefs        = np.empty(shape=(ncols, ), dtype=np.float32)
            col_lbs          = np.empty(shape=(ncols, ), dtype=np.float32)
            col_ubs          = np.empty(shape=(ncols, ), dtype=np.float32)
            col_basestats    = np.empty(shape=(ncols, ), dtype=np.int32)
            col_redcosts     = np.empty(shape=(ncols, ), dtype=np.float32)
            col_ages         = np.empty(shape=(ncols, ), dtype=np.int32)
            col_solvals      = np.empty(shape=(ncols, ), dtype=np.float32)
            col_solfracs     = np.empty(shape=(ncols, ), dtype=np.float32)
            col_sol_is_at_lb = np.empty(shape=(ncols, ), dtype=np.int32)
            col_sol_is_at_ub = np.empty(shape=(ncols, ), dtype=np.int32)
            col_incvals      = np.empty(shape=(ncols, ), dtype=np.float32)
            col_avgincvals   = np.empty(shape=(ncols, ), dtype=np.float32)
        else:
            col_types        = prev_state['col']['types']
            col_coefs        = prev_state['col']['coefs']
            col_lbs          = prev_state['col']['lbs']
            col_ubs          = prev_state['col']['ubs']
            col_basestats    = prev_state['col']['basestats']
            col_redcosts     = prev_state['col']['redcosts']
            col_ages         = prev_state['col']['ages']
            col_solvals      = prev_state['col']['solvals']
            col_solfracs     = prev_state['col']['solfracs']
            col_sol_is_at_lb = prev_state['col']['sol_is_at_lb']
            col_sol_is_at_ub = prev_state['col']['sol_is_at_ub']
            col_incvals      = prev_state['col']['incvals']
            col_avgincvals   = prev_state['col']['avgincvals']

        cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
        cdef SCIP_VAR* var
        cdef SCIP_Real lb, ub, solval
        for i in range(ncols):
            col_i = SCIPcolGetLPPos(cols[i])
            var = SCIPcolGetVar(cols[i])

            lb = SCIPcolGetLb(cols[i])
            ub = SCIPcolGetUb(cols[i])
            solval = SCIPcolGetPrimsol(cols[i])

            if not update:
                # Variable type
                col_types[col_i] = SCIPvarGetType(var)

                # Objective coefficient
                col_coefs[col_i] = SCIPcolGetObj(cols[i])

            # Lower bound
            if SCIPisInfinity(scip, REALABS(lb)):
                col_lbs[col_i] = NAN
            else:
                col_lbs[col_i] = lb

            # Upper bound
            if SCIPisInfinity(scip, REALABS(ub)):
                col_ubs[col_i] = NAN
            else:
                col_ubs[col_i] = ub

            # Basis status
            col_basestats[col_i] = SCIPcolGetBasisStatus(cols[i])

            # Reduced cost
            col_redcosts[col_i] = SCIPgetColRedcost(scip, cols[i])

            # Age
            col_ages[col_i] = cols[i].age

            # LP solution value
            col_solvals[col_i] = solval
            col_solfracs[col_i] = SCIPfeasFrac(scip, solval)
            col_sol_is_at_lb[col_i] = SCIPisEQ(scip, solval, lb)
            col_sol_is_at_ub[col_i] = SCIPisEQ(scip, solval, ub)

            # Incumbent solution value
            if sol is NULL:
                col_incvals[col_i] = NAN
                col_avgincvals[col_i] = NAN
            else:
                col_incvals[col_i] = SCIPgetSolVal(scip, sol, var)
                col_avgincvals[col_i] = SCIPvarGetAvgSol(var)


        # ROWS
        cdef int nrows = SCIPgetNLPRows(scip)
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)

        cdef np.ndarray[np.float32_t, ndim=1] row_lhss
        cdef np.ndarray[np.float32_t, ndim=1] row_rhss
        cdef np.ndarray[np.int32_t,   ndim=1] row_nnzrs
        cdef np.ndarray[np.float32_t, ndim=1] row_dualsols
        cdef np.ndarray[np.int32_t,   ndim=1] row_basestats
        cdef np.ndarray[np.int32_t,   ndim=1] row_ages
        cdef np.ndarray[np.float32_t, ndim=1] row_activities
        cdef np.ndarray[np.float32_t, ndim=1] row_objcossims
        cdef np.ndarray[np.float32_t, ndim=1] row_norms
        cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_lhs
        cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_rhs

        if not update:
            row_lhss          = np.empty(shape=(nrows, ), dtype=np.float32)
            row_rhss          = np.empty(shape=(nrows, ), dtype=np.float32)
            row_nnzrs         = np.empty(shape=(nrows, ), dtype=np.int32)
            row_dualsols      = np.empty(shape=(nrows, ), dtype=np.float32)
            row_basestats     = np.empty(shape=(nrows, ), dtype=np.int32)
            row_ages          = np.empty(shape=(nrows, ), dtype=np.int32)
            row_activities    = np.empty(shape=(nrows, ), dtype=np.float32)
            row_objcossims    = np.empty(shape=(nrows, ), dtype=np.float32)
            row_norms         = np.empty(shape=(nrows, ), dtype=np.float32)
            row_is_at_lhs     = np.empty(shape=(nrows, ), dtype=np.int32)
            row_is_at_rhs     = np.empty(shape=(nrows, ), dtype=np.int32)
            row_is_local      = np.empty(shape=(nrows, ), dtype=np.int32)
            row_is_modifiable = np.empty(shape=(nrows, ), dtype=np.int32)
            row_is_removable  = np.empty(shape=(nrows, ), dtype=np.int32)
        else:
            row_lhss          = prev_state['row']['lhss']
            row_rhss          = prev_state['row']['rhss']
            row_nnzrs         = prev_state['row']['nnzrs']
            row_dualsols      = prev_state['row']['dualsols']
            row_basestats     = prev_state['row']['basestats']
            row_ages          = prev_state['row']['ages']
            row_activities    = prev_state['row']['activities']
            row_objcossims    = prev_state['row']['objcossims']
            row_norms         = prev_state['row']['norms']
            row_is_at_lhs     = prev_state['row']['is_at_lhs']
            row_is_at_rhs     = prev_state['row']['is_at_rhs']
            row_is_local      = prev_state['row']['is_local']
            row_is_modifiable = prev_state['row']['is_modifiable']
            row_is_removable  = prev_state['row']['is_removable']

        cdef int nnzrs = 0
        cdef SCIP_Real activity, lhs, rhs, cst
        for i in range(nrows):

            # lhs <= activity + cst <= rhs
            lhs = SCIProwGetLhs(rows[i])
            rhs = SCIProwGetRhs(rows[i])
            cst = SCIProwGetConstant(rows[i])
            activity = SCIPgetRowLPActivity(scip, rows[i])  # cst is part of activity

            if not update:
                # number of coefficients
                row_nnzrs[i] = SCIProwGetNLPNonz(rows[i])
                nnzrs += row_nnzrs[i]

                # left-hand-side
                if SCIPisInfinity(scip, REALABS(lhs)):
                    row_lhss[i] = NAN
                else:
                    row_lhss[i] = lhs - cst

                # right-hand-side
                if SCIPisInfinity(scip, REALABS(rhs)):
                    row_rhss[i] = NAN
                else:
                    row_rhss[i] = rhs - cst

                # row properties
                row_is_local[i] = SCIProwIsLocal(rows[i])
                row_is_modifiable[i] = SCIProwIsModifiable(rows[i])
                row_is_removable[i] = SCIProwIsRemovable(rows[i])

                # Objective cosine similarity - inspired from SCIProwGetObjParallelism()
                SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
                prod = rows[i].sqrnorm * scip.lp.objsqrnorm
                row_objcossims[i] = rows[i].objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0

                # L2 norm
                row_norms[i] = SCIProwGetNorm(rows[i])  # cst ?

            # Dual solution
            row_dualsols[i] = SCIProwGetDualsol(rows[i])

            # Basis status
            row_basestats[i] = SCIProwGetBasisStatus(rows[i])

            # Age
            row_ages[i] = SCIProwGetAge(rows[i])

            # Activity
            row_activities[i] = activity - cst

            # Is tight
            row_is_at_lhs[i] = SCIPisEQ(scip, activity, lhs)
            row_is_at_rhs[i] = SCIPisEQ(scip, activity, rhs)


        cdef np.ndarray[np.int32_t,   ndim=1] coef_colidxs
        cdef np.ndarray[np.int32_t,   ndim=1] coef_rowidxs
        cdef np.ndarray[np.float32_t, ndim=1] coef_vals

        # Row coefficients
        if not update:
            coef_colidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
            coef_rowidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
            coef_vals    = np.empty(shape=(nnzrs, ), dtype=np.float32)
        else:
            coef_colidxs = prev_state['nzrcoef']['colidxs']
            coef_rowidxs = prev_state['nzrcoef']['rowidxs']
            coef_vals    = prev_state['nzrcoef']['vals']

        cdef SCIP_COL ** row_cols
        cdef SCIP_Real * row_vals

        if not update:
            j = 0
            for i in range(nrows):

                # coefficient indexes and values
                row_cols = SCIProwGetCols(rows[i])
                row_vals = SCIProwGetVals(rows[i])
                for k in range(row_nnzrs[i]):
                    coef_colidxs[j+k] = SCIPcolGetLPPos(row_cols[k])
                    coef_rowidxs[j+k] = i
                    coef_vals[j+k] = row_vals[k]

                j += row_nnzrs[i]


        return {
            'col': {
                'types':        col_types,
                'coefs':        col_coefs,
                'lbs':          col_lbs,
                'ubs':          col_ubs,
                'basestats':    col_basestats,
                'redcosts':     col_redcosts,
                'ages':         col_ages,
                'solvals':      col_solvals,
                'solfracs':     col_solfracs,
                'sol_is_at_lb': col_sol_is_at_lb,
                'sol_is_at_ub': col_sol_is_at_ub,
                'incvals':      col_incvals,
                'avgincvals':   col_avgincvals,
            },
            'row': {
                'lhss':          row_lhss,
                'rhss':          row_rhss,
                'nnzrs':         row_nnzrs,
                'dualsols':      row_dualsols,
                'basestats':     row_basestats,
                'ages':          row_ages,
                'activities':    row_activities,
                'objcossims':    row_objcossims,
                'norms':         row_norms,
                'is_at_lhs':     row_is_at_lhs,
                'is_at_rhs':     row_is_at_rhs,
                'is_local':      row_is_local,
                'is_modifiable': row_is_modifiable,
                'is_removable':  row_is_removable,
            },
            'nzrcoef': {
                'colidxs': coef_colidxs,
                'rowidxs': coef_rowidxs,
                'vals':    coef_vals,
            },
            'stats': {
                'nlps': SCIPgetNLPs(scip),
            },
        }

    def getDingStateRows(self):
        # basic features
        cdef np.ndarray[np.int32_t, ndim=1] cons_is_singleton
        cdef np.ndarray[np.int32_t, ndim=1] cons_is_aggregation
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_precedence
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_knapsack
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_logicor
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_general_linear
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_AND
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_OR
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_XOR
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_linking
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_cardinality
        cdef np.ndarray[np.int32_t, ndim=1]   cons_is_variable_bound
        cdef np.ndarray[np.float32_t, ndim=1] cons_lhs
        cdef np.ndarray[np.float32_t, ndim=1] cons_rhs
        cdef np.ndarray[np.int32_t, ndim=1] cons_nnzrs
        cdef np.ndarray[np.int32_t, ndim=1] cons_npos
        cdef np.ndarray[np.int32_t, ndim=1] cons_nneg

        # lp features
        cdef np.ndarray[np.float32_t, ndim=1] cons_dual_sol
        cdef np.ndarray[np.int32_t, ndim=1] cons_basis_status

        # structural features
        cdef np.ndarray[np.float32_t, ndim=1] cons_sum_abs
        cdef np.ndarray[np.float32_t, ndim=1] cons_sum_pos
        cdef np.ndarray[np.float32_t,   ndim=1] cons_sum_neg
        cdef np.ndarray[np.float32_t,   ndim=1] cons_coef_mean
        cdef np.ndarray[np.float32_t,   ndim=1] cons_coef_stdev
        cdef np.ndarray[np.float32_t,   ndim=1] cons_coef_min
        cdef np.ndarray[np.float32_t,   ndim=1] cons_coef_max

        cdef SCIP* scip = self._scip
        cdef int nrows = SCIPgetNLPRows(scip)
        
    # basic features
        cons_is_singleton                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_aggregation                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_precedence                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_knapsack                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_logicor                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_general_linear                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_AND                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_OR                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_XOR                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_linking                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_cardinality                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_is_variable_bound                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_lhs                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_rhs                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_nnzrs                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_npos                = np.empty(shape=(nrows, ), dtype=np.int32)
        cons_nneg                = np.empty(shape=(nrows, ), dtype=np.int32)

        # lp features
        cons_dual_sol                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_basis_status                = np.empty(shape=(nrows, ), dtype=np.int32)

        # structural features
        cons_sum_abs                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_sum_pos                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_sum_neg                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_coef_mean                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_coef_stdev                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_coef_min                = np.empty(shape=(nrows, ), dtype=np.float32)
        cons_coef_max                = np.empty(shape=(nrows, ), dtype=np.float32)

        # COLUMNS
        cdef int row_i, nnzrs, npos, nneg
        cdef float lhs, rhs, abs_sum_norm, pos_sum_norm, neg_sum_norm, mean, stdev, min1, max1
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)

        for row_i in range(nrows):
            row = rows[row_i]
            nnzrs = SCIProwGetNNonz(row)
            lhs = SCIProwGetLhs(row); rhs = SCIProwGetRhs(row)
            consname = bytes(SCIPconshdlrGetName(SCIProwGetOriginCons(row))).decode('UTF-8')

            cons_is_singleton[row_i] = 1 if nnzrs == 1 else 0
            cons_is_aggregation[row_i] = 1 if lhs == rhs and nnzrs > 1 else 0
            cons_is_precedence[row_i] = 1 if consname == 'cumulative' else 0
            cons_is_knapsack[row_i] = 1 if consname == 'knapsack' else 0
            cons_is_logicor[row_i] = 1 if consname == 'logicor' else 0
            cons_is_general_linear[row_i] = 1 if consname == 'linear' else 0
            cons_is_AND[row_i] = 1 if consname == 'and' else 0
            cons_is_OR[row_i] = 1 if consname == 'or' else 0
            cons_is_XOR[row_i] = 1 if consname == 'xor' else 0
            cons_is_linking[row_i] = 1 if consname == 'linking' else 0
            cons_is_cardinality[row_i] = 1 if consname == 'cardinality' else 0
            cons_is_variable_bound[row_i] = 1 if consname == 'varbound' else 0
            cons_lhs[row_i] = lhs
            cons_rhs[row_i] = rhs
            cons_nnzrs[row_i] = nnzrs

            
            vals = SCIProwGetVals(row)
            npos = 0; nneg = 0; abs_sum_norm = 0; pos_sum_norm = 0; neg_sum_norm = 0
            mean = 0; stdev = 0; min1 = 0; max1 = 0
            for i in range(nnzrs):
                abs_sum_norm += REALABS(vals[i])
                min1 = min(min1, vals[i])
                max1 = max(max1, vals[i])
                if vals[i] > 0:
                    npos += 1
                    pos_sum_norm += vals[i]
                else:
                    nneg += 1
                    neg_sum_norm += -vals[i]
                
            mean = abs_sum_norm / nnzrs if nnzrs != 0 else 0

            for i in range(nnzrs):
                stdev += (vals[i] - mean) ** 2
            stdev = SQRT(stdev / nnzrs) if nnzrs != 0 else 0

            cons_npos[row_i] = npos
            cons_nneg[row_i] = nneg
            cons_sum_abs[row_i] = abs_sum_norm
            cons_sum_pos[row_i] = pos_sum_norm
            cons_sum_neg[row_i] = neg_sum_norm
            cons_coef_mean[row_i] = mean
            cons_coef_stdev[row_i] = stdev
            cons_coef_min[row_i] = min1
            cons_coef_max[row_i] = max1

            cons_dual_sol[row_i] = SCIProwGetDualsol(row)
            stat = SCIProwGetBasisStatus(row)
            if stat == SCIP_BASESTAT_LOWER:
                cons_basis_status[row_i] = 0
            elif stat == SCIP_BASESTAT_BASIC:
                cons_basis_status[row_i] = 1
            elif stat == SCIP_BASESTAT_UPPER:
                cons_basis_status[row_i] = 2
            else:
                cons_basis_status[row_i] = 3

        return {
        # basic features
            'cons_is_singleton': cons_is_singleton,
            'cons_is_aggregation': cons_is_aggregation,
            'cons_is_precedence': cons_is_precedence,
            'cons_is_knapsack': cons_is_knapsack, 
            'cons_is_logicor': cons_is_logicor, 
            'cons_is_general_linear': cons_is_general_linear,
            'cons_is_AND': cons_is_AND,
            'cons_is_OR': cons_is_OR,
            'cons_is_XOR': cons_is_XOR,
            'cons_is_linking': cons_is_linking,
            'cons_is_cardinality': cons_is_cardinality,
            'cons_is_variable_bound': cons_is_variable_bound,
            'cons_lhs': cons_lhs,
            'cons_rhs': cons_rhs,
            'cons_nnzrs': cons_nnzrs,
            'cons_npos': cons_npos,
            'cons_nneg': cons_nneg,

            # lp features
            'cons_dual_sol': cons_dual_sol,
            'cons_basis_status': cons_basis_status,

            # structural features
            'cons_sum_abs': cons_sum_abs,
            'cons_sum_pos': cons_sum_pos,
            'cons_sum_neg': cons_sum_neg,
            'cons_coef_mean': cons_coef_mean,
            'cons_coef_stdev': cons_coef_stdev,
            'cons_coef_min': cons_coef_min,
            'cons_coef_max': cons_coef_max    
       }
    

    def getDingStateLPgraph(self):
        cdef SCIP* scip = self._scip
        cdef int row_i, col_i, i
        
        cdef int ncols = SCIPgetNLPCols(scip)
        cdef int nrows = SCIPgetNLPRows(scip)

        cdef np.ndarray[np.float32_t, ndim=2] vc
        cdef np.ndarray[np.float32_t, ndim=1] vo
        cdef np.ndarray[np.float32_t, ndim=1] co
        vc = np.zeros(shape=(nrows, ncols), dtype=np.float32)
        vo = np.zeros(shape=(ncols, ), dtype=np.float32)
        co = np.zeros(shape=(nrows, ), dtype=np.float32)

        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        for col_i in range(ncols):
            # vo adj
            vo[col_i] = SCIPcolGetObj(cols[col_i])
        
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef int ncols_row
        cdef SCIP_COL** cols_row
        cdef SCIP_Real* row_coefs
        for row_i in range(nrows):
            # co adj
            co[row_i] = SCIProwGetRhs(rows[row_i])
        
            # vc adj
            ncols_row = SCIProwGetNLPNonz(rows[row_i])
            cols_row = SCIProwGetCols(rows[row_i])
            row_coefs = SCIProwGetVals(rows[row_i])
            for i in range(ncols_row):
                col_i = SCIPcolGetLPPos(cols_row[i])
                vc[row_i, col_i] = row_coefs[i]

        return vc, vo, co


    def getDingStateCols(self):

        cdef np.ndarray[np.int32_t,   ndim=1] col_type_binary
        cdef np.ndarray[np.int32_t,   ndim=1] col_type_int
        cdef np.ndarray[np.float32_t, ndim=1] col_coefs
        cdef np.ndarray[np.float32_t, ndim=1] col_coefs_pos
        cdef np.ndarray[np.float32_t, ndim=1] col_coefs_neg
        cdef np.ndarray[np.int32_t, ndim=1]   col_nnzrs
        cdef np.ndarray[np.int32_t, ndim=1]   col_nup_locks
        cdef np.ndarray[np.int32_t, ndim=1]   col_ndown_locks

        cdef np.ndarray[np.float32_t, ndim=1] col_solvals
        cdef np.ndarray[np.float32_t, ndim=1] col_sol_frac_up
        cdef np.ndarray[np.float32_t, ndim=1] col_sol_frac_down
        cdef np.ndarray[np.float32_t, ndim=1] col_sol_isfrac

        cdef np.ndarray[np.float32_t, ndim=1] col_ps_up
        cdef np.ndarray[np.float32_t, ndim=1] col_ps_down
        cdef np.ndarray[np.float32_t, ndim=1] col_ps_ratio
        cdef np.ndarray[np.float32_t, ndim=1] col_ps_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_ps_product
        cdef np.ndarray[np.float32_t, ndim=1] col_lbs
        cdef np.ndarray[np.float32_t, ndim=1] col_ubs
        cdef np.ndarray[np.float32_t, ndim=1] col_red_costs

        cdef np.ndarray[np.float32_t, ndim=1] col_cdeg_mean
        cdef np.ndarray[np.float32_t, ndim=1] col_cdeg_std
        cdef np.ndarray[np.int32_t, ndim=1]   col_cdeg_min
        cdef np.ndarray[np.int32_t, ndim=1]   col_cdeg_max

        cdef np.ndarray[np.float32_t, ndim=1] col_prhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] col_prhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] col_nrhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] col_nrhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] col_plhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] col_plhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] col_nlhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] col_nlhs_ratio_min

        cdef np.ndarray[np.float32_t, ndim=1] col_pcoefs_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_pcoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] col_pcoefs_std
        cdef np.ndarray[np.float32_t, ndim=1] col_pcoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] col_pcoefs_max
        cdef np.ndarray[np.float32_t, ndim=1] col_ncoefs_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_ncoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] col_ncoefs_std
        cdef np.ndarray[np.float32_t, ndim=1] col_ncoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] col_ncoefs_max

        cdef np.ndarray[np.float32_t, ndim=1] col_coef_sum1_unit_weight
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_mean1_unit_weight
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_std1_unit_weight
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_max1_unit_weight
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_min1_unit_weight
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_sum2_inverse_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_mean2_inverse_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_std2_inverse_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_max2_inverse_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_min2_inverse_sum
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_sum3_dual_cost
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_mean3_dual_cost
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_std3_dual_cost
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_max3_dual_cost
        cdef np.ndarray[np.float32_t, ndim=1] col_coef_min3_dual_cost

        cdef SCIP* scip = self._scip
        cdef int i, j, k, col_i
        cdef SCIP_Real sim, prod
        cdef int ncols = SCIPgetNLPCols(scip)
        
        col_type_binary                = np.empty(shape=(ncols, ), dtype=np.int32)
        col_type_int                = np.empty(shape=(ncols, ), dtype=np.int32)
        col_coefs                = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coefs_pos            = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coefs_neg            = np.empty(shape=(ncols, ), dtype=np.float32)
        col_nnzrs                = np.empty(shape=(ncols, ), dtype=np.int32)
        col_nup_locks            = np.empty(shape=(ncols, ), dtype=np.int32)
        col_ndown_locks          = np.empty(shape=(ncols, ), dtype=np.int32)

        col_solvals              = np.empty(shape=(ncols, ), dtype=np.float32)
        col_sol_frac_up              = np.empty(shape=(ncols, ), dtype=np.float32)
        col_sol_frac_down            = np.empty(shape=(ncols, ), dtype=np.float32)
        col_sol_isfrac        = np.empty(shape=(ncols, ), dtype=np.float32)
        col_red_costs            = np.empty(shape=(ncols, ), dtype=np.float32)
        col_lbs                  = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ubs                  = np.empty(shape=(ncols, ), dtype=np.float32)

        col_ps_up                = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ps_down              = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ps_ratio             = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ps_sum               = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ps_product           = np.empty(shape=(ncols, ), dtype=np.float32)

        col_cdeg_mean           = np.empty(shape=(ncols, ), dtype=np.float32)
        col_cdeg_std            = np.empty(shape=(ncols, ), dtype=np.float32)
        col_cdeg_min            = np.empty(shape=(ncols, ), dtype=np.int32)
        col_cdeg_max            = np.empty(shape=(ncols, ), dtype=np.int32)

        col_prhs_ratio_max      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_prhs_ratio_min      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_nrhs_ratio_max      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_nrhs_ratio_min      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_plhs_ratio_max      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_plhs_ratio_min      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_nlhs_ratio_max      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_nlhs_ratio_min      = np.empty(shape=(ncols, ), dtype=np.float32)
    

        col_pcoefs_sum      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_pcoefs_mean     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_pcoefs_std      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_pcoefs_min      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_pcoefs_max      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ncoefs_sum      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ncoefs_mean     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ncoefs_std      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ncoefs_min      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ncoefs_max      = np.empty(shape=(ncols, ), dtype=np.float32)

        col_coef_sum1_unit_weight          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_mean1_unit_weight         = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_std1_unit_weight          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_max1_unit_weight          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_min1_unit_weight          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_sum2_inverse_sum          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_mean2_inverse_sum         = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_std2_inverse_sum          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_max2_inverse_sum          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_min2_inverse_sum          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_sum3_dual_cost          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_mean3_dual_cost         = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_std3_dual_cost          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_max3_dual_cost          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_coef_min3_dual_cost          = np.empty(shape=(ncols, ), dtype=np.float32)

        # COLUMNS
        cdef SCIP_COL** cols = SCIPgetLPCols(scip)

        cdef SCIP_ROW** neighbors
        cdef SCIP_Real* nonzero_coefs_raw
        cdef SCIP_Real* all_coefs_raw
        cdef SCIP_Real  lhs, rhs, coef
        cdef SCIP_VAR* var
        cdef SCIP_COL* col
        cdef int neighbor_index, cdeg_max, cdeg_min, cdeg, nb_neighbors
        cdef float cdeg_mean, cdeg_var
        cdef int pcoefs_count, ncoefs_count
        cdef float pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max
        cdef float ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max

        for i in range(ncols):
            col = cols[i]
            col_i = SCIPcolGetIndex(col)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)
            var = SCIPcolGetVar(cols[i])

            ##### basic features #####

            # Variable type
            col_type_binary[col_i] = SCIPvarIsBinary(var)
            col_type_int[col_i] = SCIPvarIsIntegral(var)

            # Objective coefficient
            col_coefs[col_i] = SCIPcolGetObj(cols[i])
            col_coefs_pos[col_i] = max(col_coefs[col_i], 0)
            col_coefs_neg[col_i] = min(col_coefs[col_i], 0)

            # nonzeros for col in constraints
            col_nnzrs[col_i] = SCIPcolGetNNonz(col)

            # locks
            col_nup_locks[col_i] = SCIPvarGetNLocksUp(var)
            col_ndown_locks[col_i] = SCIPvarGetNLocksDown(var)

            ##### lp features #####
            solval = SCIPcolGetPrimsol(cols[i])
            col_solvals[col_i] = solval
            col_sol_frac_up[col_i] = math.ceil(solval) - solval
            col_sol_frac_down[col_i] = solval - math.floor(solval)
            col_sol_isfrac[col_i] = SCIPfeasFrac(scip, solval)

            # Global bounds
            col_lbs[col_i] = SCIPcolGetLb(cols[i])
            col_ubs[col_i] = SCIPcolGetUb(cols[i])

            # reduced cost
            col_red_costs[col_i] = SCIPgetColRedcost(scip, cols[i])

            # Stats. for constraint degrees (4)
            #   The degree of a constraint is the number of variables that participate in it. A variable may
            #   participate in multiple constraints, and statistics over those constraints’ degrees are used.
            #   The constraint degree is computed on the root LP (mean, stdev., min, max)
            cdeg_var, cdeg_mean, cdeg_min, cdeg_max = 0, 0, 0, 0
            if nb_neighbors > 0:
                for neighbor_index in range(nb_neighbors):
                    cdeg = SCIProwGetNNonz(neighbors[neighbor_index])
                    cdeg_mean += cdeg
                    cdeg_max = cdeg if neighbor_index == 0 else max(cdeg_max, cdeg)
                    cdeg_min = cdeg if neighbor_index == 0 else min(cdeg_min, cdeg)
                cdeg_mean /= nb_neighbors
                for neighbor_index in range(nb_neighbors):
                    cdeg_var += (cdeg - cdeg_mean)**2
                cdeg_var /= nb_neighbors

            col_cdeg_mean[col_i] = cdeg_mean
            col_cdeg_std[col_i] =  math.sqrt(cdeg_var)
            col_cdeg_min[col_i] = cdeg_min
            col_cdeg_max[col_i] = cdeg_max

            # Min/max for ratios of constraint coeffs. to RHS/LHS (8)
            prhs_ratio_max, prhs_ratio_min = -1, 1
            nrhs_ratio_max, nrhs_ratio_min = -1, 1
            plhs_ratio_max, plhs_ratio_min = -1, 1
            nlhs_ratio_max, nlhs_ratio_min = -1, 1
            for neighbor_index in range(nb_neighbors):
                coef = nonzero_coefs_raw[neighbor_index]
                rhs = SCIProwGetRhs(neighbors[neighbor_index])
                lhs = SCIProwGetLhs(neighbors[neighbor_index])
                if not SCIPisInfinity(scip, REALABS(rhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(rhs))
                    if rhs >= 0:
                        prhs_ratio_max = max(prhs_ratio_max, value)
                        prhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)

                if not SCIPisInfinity(scip, REALABS(lhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(lhs))
                    if lhs >= 0:
                        prhs_ratio_max = max(prhs_ratio_max, value)
                        prhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)
            col_prhs_ratio_max[col_i] = prhs_ratio_max
            col_prhs_ratio_min[col_i] = prhs_ratio_min
            col_nrhs_ratio_max[col_i] = nrhs_ratio_max
            col_nrhs_ratio_min[col_i] = nrhs_ratio_min

            col_plhs_ratio_max[col_i] = plhs_ratio_max
            col_plhs_ratio_min[col_i] = plhs_ratio_min
            col_nlhs_ratio_max[col_i] = nlhs_ratio_max
            col_nlhs_ratio_min[col_i] = nlhs_ratio_min

            
            # Stats. for constraint coeffs. (10)
            #   A variable’s positive (negative) coefficients in the constraints it participates in
            #   (count, mean, stdev., min, max)
            pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max = 0, 0, 0, 0.
            ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max = 0, 0, 0, 0.
            pcoefs_count, ncoefs_count = 0, 0
            for neighbor_index in range(nb_neighbors):
                coef = nonzero_coefs_raw[neighbor_index]
                if coef > 0:
                    pcoefs_count += 1
                    pcoefs_mean = coef
                    pcoefs_min = coef if pcoefs_count == 1 else min(pcoefs_min, coef)
                    pcoefs_max = coef if pcoefs_count == 1 else max(pcoefs_max, coef)
                if coef < 0:
                    ncoefs_count += 1
                    ncoefs_mean += coef
                    ncoefs_min = coef if ncoefs_count == 1 else min(ncoefs_min, coef)
                    ncoefs_max = coef if ncoefs_count == 1 else max(ncoefs_max, coef)
            if pcoefs_count > 0:
                pcoefs_mean /= pcoefs_count
            if ncoefs_count > 0:
                ncoefs_mean /= ncoefs_count
            for neighbor_index in range(nb_neighbors):
                coef = nonzero_coefs_raw[neighbor_index]
                if coef > 0:
                    pcoefs_var += (coef - pcoefs_mean)**2
                if coef < 0:
                    ncoefs_var += (coef - ncoefs_mean)**2
            if pcoefs_count > 0:
                pcoefs_var /= pcoefs_count
            if ncoefs_count > 0:
                ncoefs_var /= ncoefs_count
            col_pcoefs_sum[col_i]   = pcoefs_count
            col_pcoefs_mean[col_i]  = pcoefs_mean
            col_pcoefs_std[col_i]   = math.sqrt(pcoefs_var)
            col_pcoefs_min[col_i]   = pcoefs_min
            col_pcoefs_max[col_i]   = pcoefs_max
            col_ncoefs_sum[col_i]   = ncoefs_count
            col_ncoefs_mean[col_i]  = ncoefs_mean
            col_ncoefs_std[col_i]   = math.sqrt(ncoefs_var)
            col_ncoefs_min[col_i]   = ncoefs_min
            col_ncoefs_max[col_i]   = ncoefs_max

            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)

        # --------------
        # Stats. for constraint coefficients (3 * 5)
        #   An active constraint at a node LP is one which is binding with equality at the optimum.
        #   We consider 3 weighting schemes for an active constraint: unit weight, inverse of the
        #   sum of the coefficients of all variables in constraint, dual cost of the constraint. Given the absolute
        #   value of the coefficients of xj in the active constraints, we compute the sum, mean, stdev.,
        #   max. and min. of those values, for each of the weighting schemes. We also compute the weighted
        #   number of active constraints that xj is in, with the same 4 weightings
        cdef int row_index
        cdef int nrows = SCIPgetNLPRows(scip)
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef float constraint_sum, abs_coef
        cdef SCIP_COL** neighbor_columns
        cdef int neighbor_var_index, candidate_index
        cdef int count
        cdef float cons_sum1, cons_mean1, cons_var1, cons_max1, cons_min1
        cdef float cons_sum2, cons_mean2, cons_var2, cons_max2, cons_min2
        cdef float cons_sum3, cons_mean3, cons_var3, cons_max3, cons_min3
        cdef np.ndarray[np.float32_t, ndim=1] cons_w1, cons_w2, cons_w3, act_cons_w4

        cons_w1 = np.zeros(shape=(nrows, ), dtype=np.float32)
        cons_w2 = np.zeros(shape=(nrows, ), dtype=np.float32)
        cons_w3 = np.zeros(shape=(nrows, ), dtype=np.float32)

        for row_index in range(nrows):
            row = rows[row_index]

            neighbor_columns = SCIProwGetCols(row)
            neighbor_ncolumns = SCIProwGetNNonz(row)
            neighbor_columns_values = SCIProwGetVals(row)

            # weight no. 1
            # unit weight
            cons_w1[row_index] = 1

            # weight no. 2
            # inverse of the sum of the coefficients of all variables in constraint
            constraint_sum = 0
            for neighbor_column_index in range(neighbor_ncolumns):
                constraint_sum += REALABS(neighbor_columns_values[neighbor_column_index])
            cons_w2[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum

            # weight no. 3
            # dual cost of the constraint
            cons_w3[row_index] = REALABS(SCIProwGetDualsol(row))


        for i in range(ncols):
            col = cols[i]
            col_i = SCIPcolGetIndex(col)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)
            var = SCIPcolGetVar(cols[i])

            cons_sum1, cons_mean1, cons_var1, cons_max1, cons_min1 = 0, 0, 0, 0, 0
            cons_sum2, cons_mean2, cons_var2, cons_max2, cons_min2 = 0, 0, 0, 0, 0
            cons_sum3, cons_mean3, cons_var3, cons_max3, cons_min3 = 0, 0, 0, 0, 0
            count = 0
            for neighbor_index in range(nb_neighbors):
                count += 1
                neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                value = cons_w1[neighbor_row_index] * abs_coef
                cons_sum1 += value
                cons_max1 = value if count == 1 else max(cons_max1, value)
                cons_min1 = value if count == 1 else min(cons_min1, value)

                value = cons_w2[neighbor_row_index] * abs_coef
                cons_sum2 += value
                cons_max2 = value if count == 1 else max(cons_max2, value)
                cons_min2 = value if count == 1 else min(cons_min2, value)

                value = cons_w3[neighbor_row_index] * abs_coef
                cons_sum3 += value
                cons_max3 = value if count == 1 else max(cons_max3, value)
                cons_min3 = value if count == 1 else min(cons_min3, value)

            if count > 0:
                cons_mean1 = cons_sum1 / count
                cons_mean2 = cons_sum2 / count
                cons_mean3 = cons_sum3 / count

                for neighbor_index in range(nb_neighbors):

                    neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                    abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                    value = cons_w1[neighbor_row_index] * abs_coef
                    cons_var1 += (value - cons_mean1)**2

                    value = cons_w2[neighbor_row_index] * abs_coef
                    cons_var2 += (value - cons_mean2)**2

                    value = cons_w3[neighbor_row_index] * abs_coef
                    cons_var3 += (value - cons_mean3)**2

                cons_var1 /= count
                cons_var2 /= count
                cons_var3 /= count

            col_coef_sum1_unit_weight[col_i]  = cons_sum1
            col_coef_sum2_inverse_sum[col_i]  = cons_sum2
            col_coef_sum3_dual_cost[col_i]  = cons_sum3
            col_coef_mean1_unit_weight[col_i] = cons_mean1
            col_coef_mean2_inverse_sum[col_i] = cons_mean2
            col_coef_mean3_dual_cost[col_i] = cons_mean3
            col_coef_max1_unit_weight[col_i]  = cons_max1
            col_coef_max2_inverse_sum[col_i]  = cons_max2
            col_coef_max3_dual_cost[col_i]  = cons_max3
            col_coef_min1_unit_weight[col_i]  = cons_min1
            col_coef_min2_inverse_sum[col_i]  = cons_min2
            col_coef_min3_dual_cost[col_i]  = cons_min3
            col_coef_std1_unit_weight[col_i]  = math.sqrt(cons_var1)
            col_coef_std2_inverse_sum[col_i]  = math.sqrt(cons_var2)
            col_coef_std3_dual_cost[col_i]  = math.sqrt(cons_var3)


        return {
            # basic (8)
            'col_type_binary':          col_type_binary,
            'col_type_int':             col_type_int,
            'col_coefs':                col_coefs,
            'col_coefs_pos':            col_coefs_pos,
            'col_coefs_neg':            col_coefs_neg,
            'col_nnzrs':                col_nnzrs,
            'col_nup_locks':            col_nup_locks,
            'col_ndown_locks':          col_ndown_locks,

            # LP (12)
            'col_solvals':              col_solvals,
            'col_sol_frac_up':          col_sol_frac_up,
            'col_sol_frac_down':        col_sol_frac_down,
            'col_sol_isfrac':           col_sol_isfrac,
            'col_red_costs':            col_red_costs,
            'col_lbs':                  col_lbs,
            'col_ubs':                  col_ubs,
            'col_ps_up':                col_ps_up,
            'col_ps_down':              col_ps_down,
            'col_ps_ratio':             col_ps_ratio,
            'col_ps_sum':               col_ps_sum,
            'col_ps_product':           col_ps_product,

            # Structure (4 + 8 + 10 + 15)
            'col_cdeg_mean':            col_cdeg_mean,
            'col_cdeg_std':             col_cdeg_std,
            'col_cdeg_min':             col_cdeg_min,
            'col_cdeg_max':             col_cdeg_max,
            'col_prhs_ratio_max':            col_prhs_ratio_max,
            'col_prhs_ratio_min':            col_prhs_ratio_min,
            'col_nrhs_ratio_max':            col_nrhs_ratio_max,
            'col_nrhs_ratio_min':            col_nrhs_ratio_min,
            'col_plhs_ratio_max':            col_plhs_ratio_max,
            'col_plhs_ratio_min':            col_plhs_ratio_min,
            'col_nlhs_ratio_max':            col_nlhs_ratio_max,
            'col_nlhs_ratio_min':            col_nlhs_ratio_min,
            'col_pcoefs_sum':            col_pcoefs_sum,
            'col_pcoefs_mean':            col_pcoefs_mean,
            'col_pcoefs_std':            col_pcoefs_std,
            'col_pcoefs_min':            col_pcoefs_min,
            'col_pcoefs_max':            col_pcoefs_max,
            'col_ncoefs_sum':            col_ncoefs_sum,
            'col_ncoefs_mean':            col_ncoefs_mean,
            'col_ncoefs_std':            col_ncoefs_std,
            'col_ncoefs_min':            col_ncoefs_min,
            'col_ncoefs_max':            col_ncoefs_max,
            'col_coef_sum1_unit_weight':            col_coef_sum1_unit_weight,
            'col_coef_mean1_unit_weight':            col_coef_mean1_unit_weight,
            'col_coef_std1_unit_weight':            col_coef_std1_unit_weight,
            'col_coef_max1_unit_weight':            col_coef_max1_unit_weight,
            'col_coef_min1_unit_weight':            col_coef_min1_unit_weight,
            'col_coef_sum2_inverse_sum':            col_coef_sum2_inverse_sum,
            'col_coef_mean2_inverse_sum':            col_coef_mean2_inverse_sum,
            'col_coef_std2_inverse_sum':            col_coef_std2_inverse_sum,
            'col_coef_max2_inverse_sum':            col_coef_max2_inverse_sum,
            'col_coef_min2_inverse_sum':            col_coef_min2_inverse_sum,
            'col_coef_sum3_dual_cost':            col_coef_sum3_dual_cost,
            'col_coef_mean3_dual_cost':            col_coef_mean3_dual_cost,
            'col_coef_std3_dual_cost':            col_coef_std3_dual_cost,
            'col_coef_max3_dual_cost':            col_coef_max3_dual_cost,
            'col_coef_min3_dual_cost':            col_coef_min3_dual_cost,
        }

    def getKhalilState(self, root_info, candidates):
        cdef SCIP* scip = self._scip

        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs
        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs_pos
        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs_neg
        cdef np.ndarray[np.int32_t, ndim=1]   cand_nnzrs
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_cdeg_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_cdeg_var
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_cdeg_min
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_cdeg_max
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_pcoefs_count
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_var
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_max
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_ncoefs_count
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_var
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_slack
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_up
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_down
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_sum
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_product
        cdef np.ndarray[np.float32_t, ndim=1] cand_nb_up_infeas
        cdef np.ndarray[np.float32_t, ndim=1] cand_nb_down_infeas
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_var
        cdef np.ndarray[np.int32_t, ndim=1]   cand_cdeg_min
        cdef np.ndarray[np.int32_t, ndim=1]   cand_cdeg_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_mean_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_min_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_max_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_prhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_prhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_nrhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_nrhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pp_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pp_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pn_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pn_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_np_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_np_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_nn_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_nn_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb4

        cdef int ncands = len(candidates)

        cand_coefs               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_coefs_pos           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_coefs_neg           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nnzrs               = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_cdeg_mean      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_cdeg_var       = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_cdeg_min       = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_cdeg_max       = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_pcoefs_count   = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_pcoefs_var     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_mean    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_min     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_max     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_count   = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_ncoefs_mean    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_var     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_min     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_max     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_solfracs            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_slack               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_up               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_down             = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_ratio            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_sum              = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_product          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_frac_up_infeas      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_frac_down_infeas    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nb_up_infeas        = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nb_down_infeas      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_mean           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_var            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_min            = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_cdeg_max            = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_cdeg_mean_ratio     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_min_ratio      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_max_ratio      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_prhs_ratio_max      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_prhs_ratio_min      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nrhs_ratio_max      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nrhs_ratio_min      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pp_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pp_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pn_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pn_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_np_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_np_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_nn_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_nn_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean1         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean2         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean3         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean4         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb1           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb2           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb3           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb4           = np.empty(shape=(ncands, ), dtype=np.float32)

        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)
        cdef int i, cand_i, col_i

        # Static
        # ------
        cdef SCIP_ROW** neighbors
        cdef SCIP_Real* nonzero_coefs_raw
        cdef SCIP_Real* all_coefs_raw
        cdef SCIP_Real activity, lhs, rhs, coef
        cdef SCIP_VAR* var
        cdef SCIP_COL* col
        cdef int neighbor_index, cdeg_max, cdeg_min, cdeg, nb_neighbors
        cdef float cdeg_mean, cdeg_var
        cdef int pcoefs_count, ncoefs_count
        cdef float pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max
        cdef float ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max

        # if at root node, extract root information
        if SCIPgetNNodes(scip) == 1:
            root_info['col'] = {}
            root_info['col']['coefs']               = {}
            root_info['col']['coefs_pos']           = {}
            root_info['col']['coefs_neg']           = {}
            root_info['col']['nnzrs']               = {}
            root_info['col']['root_cdeg_mean']      = {}
            root_info['col']['root_cdeg_var']       = {}
            root_info['col']['root_cdeg_min']       = {}
            root_info['col']['root_cdeg_max']       = {}
            root_info['col']['root_pcoefs_count']   = {}
            root_info['col']['root_pcoefs_var']     = {}
            root_info['col']['root_pcoefs_mean']    = {}
            root_info['col']['root_pcoefs_min']     = {}
            root_info['col']['root_pcoefs_max']     = {}
            root_info['col']['root_ncoefs_count']   = {}
            root_info['col']['root_ncoefs_mean']    = {}
            root_info['col']['root_ncoefs_var']     = {}
            root_info['col']['root_ncoefs_min']     = {}
            root_info['col']['root_ncoefs_max']     = {}
            for i in range(ncols):
                col = cols[i]
                col_i = SCIPcolGetIndex(col)
                neighbors = SCIPcolGetRows(col)
                nb_neighbors = SCIPcolGetNNonz(col)
                nonzero_coefs_raw = SCIPcolGetVals(col)

                # Objective function coeffs. (3)
                #   Value of the coefficient (raw, positive only, negative only)
                root_info['col']['coefs'][col_i]     = SCIPcolGetObj(col)
                root_info['col']['coefs_pos'][col_i] = max(root_info['col']['coefs'][col_i], 0)
                root_info['col']['coefs_neg'][col_i] = min(root_info['col']['coefs'][col_i], 0)

                # Num. constraints (1)
                #   Number of constraints that the variable participates in (with a non-zero coefficient)
                root_info['col']['nnzrs'][col_i] = nb_neighbors

                # Stats. for constraint degrees (4)
                #   The degree of a constraint is the number of variables that participate in it. A variable may
                #   participate in multiple constraints, and statistics over those constraints’ degrees are used.
                #   The constraint degree is computed on the root LP (mean, stdev., min, max)
                cdeg_var, cdeg_mean, cdeg_min, cdeg_max = 0, 0, 0, 0
                if nb_neighbors > 0:
                    for neighbor_index in range(nb_neighbors):
                        cdeg = SCIProwGetNNonz(neighbors[neighbor_index])
                        cdeg_mean += cdeg
                        cdeg_max = cdeg if neighbor_index == 0 else max(cdeg_max, cdeg)
                        cdeg_min = cdeg if neighbor_index == 0 else min(cdeg_min, cdeg)
                    cdeg_mean /= nb_neighbors
                    for neighbor_index in range(nb_neighbors):
                        cdeg_var += (cdeg - cdeg_mean)**2
                    cdeg_var /= nb_neighbors
                root_info['col']['root_cdeg_mean'][col_i] = cdeg_mean
                root_info['col']['root_cdeg_var'][col_i] = cdeg_var
                root_info['col']['root_cdeg_min'][col_i] = cdeg_min
                root_info['col']['root_cdeg_max'][col_i] = cdeg_max

                # Stats. for constraint coeffs. (10)
                #   A variable’s positive (negative) coefficients in the constraints it participates in
                #   (count, mean, stdev., min, max)
                pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max = 0, 0, 0, 0.
                ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max = 0, 0, 0, 0.
                pcoefs_count, ncoefs_count = 0, 0
                for neighbor_index in range(nb_neighbors):
                    coef = nonzero_coefs_raw[neighbor_index]
                    if coef > 0:
                        pcoefs_count += 1
                        pcoefs_mean = coef
                        pcoefs_min = coef if pcoefs_count == 1 else min(pcoefs_min, coef)
                        pcoefs_max = coef if pcoefs_count == 1 else max(pcoefs_max, coef)
                    if coef < 0:
                        ncoefs_count += 1
                        ncoefs_mean += coef
                        ncoefs_min = coef if ncoefs_count == 1 else min(ncoefs_min, coef)
                        ncoefs_max = coef if ncoefs_count == 1 else max(ncoefs_max, coef)
                if pcoefs_count > 0:
                    pcoefs_mean /= pcoefs_count
                if ncoefs_count > 0:
                    ncoefs_mean /= ncoefs_count
                for neighbor_index in range(nb_neighbors):
                    coef = nonzero_coefs_raw[neighbor_index]
                    if coef > 0:
                        pcoefs_var += (coef - pcoefs_mean)**2
                    if coef < 0:
                        ncoefs_var += (coef - ncoefs_mean)**2
                if pcoefs_count > 0:
                    pcoefs_var /= pcoefs_count
                if ncoefs_count > 0:
                    ncoefs_var /= ncoefs_count
                root_info['col']['root_pcoefs_count'][col_i] = pcoefs_count
                root_info['col']['root_pcoefs_mean'][col_i]  = pcoefs_mean
                root_info['col']['root_pcoefs_var'][col_i]   = pcoefs_var
                root_info['col']['root_pcoefs_min'][col_i]   = pcoefs_min
                root_info['col']['root_pcoefs_max'][col_i]   = pcoefs_max
                root_info['col']['root_ncoefs_count'][col_i] = ncoefs_count
                root_info['col']['root_ncoefs_mean'][col_i]  = ncoefs_mean
                root_info['col']['root_ncoefs_var'][col_i]   = ncoefs_var
                root_info['col']['root_ncoefs_min'][col_i]   = ncoefs_min
                root_info['col']['root_ncoefs_max'][col_i]   = ncoefs_max

        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            col_i = SCIPcolGetIndex(col)
            cand_coefs[cand_i]             = root_info['col']['coefs'][col_i]
            cand_coefs_pos[cand_i]         = root_info['col']['coefs_pos'][col_i]
            cand_coefs_neg[cand_i]         = root_info['col']['coefs_neg'][col_i]
            cand_nnzrs[cand_i]             = root_info['col']['nnzrs'][col_i]
            cand_root_cdeg_mean[cand_i]    = root_info['col']['root_cdeg_mean'][col_i]
            cand_root_cdeg_var[cand_i]     = root_info['col']['root_cdeg_var'][col_i]
            cand_root_cdeg_min[cand_i]     = root_info['col']['root_cdeg_min'][col_i]
            cand_root_cdeg_max[cand_i]     = root_info['col']['root_cdeg_max'][col_i]
            cand_root_pcoefs_count[cand_i] = root_info['col']['root_pcoefs_count'][col_i]
            cand_root_pcoefs_mean[cand_i]  = root_info['col']['root_pcoefs_mean'][col_i]
            cand_root_pcoefs_var[cand_i]   = root_info['col']['root_pcoefs_var'][col_i]
            cand_root_pcoefs_min[cand_i]   = root_info['col']['root_pcoefs_min'][col_i]
            cand_root_pcoefs_max[cand_i]   = root_info['col']['root_pcoefs_max'][col_i]
            cand_root_ncoefs_count[cand_i] = root_info['col']['root_ncoefs_count'][col_i]
            cand_root_ncoefs_mean[cand_i]  = root_info['col']['root_ncoefs_mean'][col_i]
            cand_root_ncoefs_var[cand_i]   = root_info['col']['root_ncoefs_var'][col_i]
            cand_root_ncoefs_min[cand_i]   = root_info['col']['root_ncoefs_min'][col_i]
            cand_root_ncoefs_max[cand_i]   = root_info['col']['root_ncoefs_max'][col_i]

        # Simple dynamic
        # --------------
        cdef int neighbor_column_index, neighbor_ncolumns
        cdef float solval, pos_coef_sum, neg_coef_sum, neighbor_coef
        cdef float ota_pp_max, ota_pp_min, ota_pn_max, ota_pn_min
        cdef float ota_np_max, ota_np_min, ota_nn_max, ota_nn_min
        cdef float prhs_ratio_max, prhs_ratio_min
        cdef float nrhs_ratio_max, nrhs_ratio_min
        cdef float value, pratio, nratio
        cdef SCIP_VAR* neighbor_var
        cdef SCIP_Real* neighbor_columns_values
        cdef int nbranchings
        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)

            # Slack and ceil distances (2)
            #   min{xij−floor(xij),ceil(xij) −xij} and ceil(xij) −xij
            solval = SCIPcolGetPrimsol(col)
            cand_solfracs[cand_i] = SCIPfeasFrac(scip, solval)
            cand_slack[cand_i] = min(cand_solfracs[cand_i], 1-cand_solfracs[cand_i])

            # Pseudocosts (5)
            #   Upwards and downwards values, and their corresponding ratio, sum and product,
            #   weighted by the fractionality of xj
            cand_ps_up[cand_i] = SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS)
            cand_ps_down[cand_i] = SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS)
            cand_ps_sum[cand_i] = cand_ps_up[cand_i] + cand_ps_down[cand_i]
            cand_ps_ratio[cand_i] = 0 if cand_ps_up[cand_i] == 0 else cand_ps_up[cand_i] / cand_ps_sum[cand_i]
            cand_ps_product[cand_i] = cand_ps_up[cand_i] * cand_ps_down[cand_i]

            # Infeasibility statistics (4)
            #   Number and fraction of nodes for which applying SB to variable xj led to one (two)
            #   infeasible children (during data collection)
            # N.B. replaced by left, right infeasibility
            cand_nb_up_infeas[cand_i]   = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS)
            cand_nb_down_infeas[cand_i] = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS)
            nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS)
            cand_frac_up_infeas[cand_i]   = 0 if nbranchings == 0 else cand_nb_up_infeas[cand_i] / nbranchings
            nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS)
            cand_frac_down_infeas[cand_i] = 0 if nbranchings == 0 else cand_nb_down_infeas[cand_i] / nbranchings

            # Stats. for constraint degrees (7)
            #   A dynamic variant of the static version above. Here, the constraint degrees are
            #   on the current node’s LP.The ratios of the static mean, maximum and minimum to
            #   their dynamic counterparts are also features
            cdeg_var, cdeg_mean, cdeg_min, cdeg_max = 0, 0, 0, 0
            if nb_neighbors > 0:
                for neighbor_index in range(nb_neighbors):
                    cdeg = SCIProwGetNLPNonz(neighbors[neighbor_index])
                    cdeg_mean += cdeg
                    cdeg_max = cdeg if neighbor_index == 0 else max(cdeg_max, cdeg)
                    cdeg_min = cdeg if neighbor_index == 0 else min(cdeg_min, cdeg)
                cdeg_mean /= nb_neighbors
                for neighbor_index in range(nb_neighbors):
                    cdeg = SCIProwGetNLPNonz(neighbors[neighbor_index])
                    cdeg_var += (cdeg - cdeg_mean)**2
                cdeg_var /= nb_neighbors
            cand_cdeg_mean[cand_i] = cdeg_mean
            cand_cdeg_var[cand_i]  = cdeg_var
            cand_cdeg_min[cand_i]  = cdeg_min
            cand_cdeg_max[cand_i]  = cdeg_max
            cand_cdeg_mean_ratio[cand_i] = 0 if cdeg_mean == 0 else cdeg_mean / (cand_root_cdeg_mean[cand_i] + cdeg_mean)
            cand_cdeg_min_ratio[cand_i]  = 0 if cdeg_min == 0 else cdeg_min / (cand_root_cdeg_min[cand_i] + cdeg_min)
            cand_cdeg_max_ratio[cand_i]  = 0 if cdeg_max == 0 else cdeg_max / (cand_root_cdeg_max[cand_i] + cdeg_max)

            # Min/max for ratios of constraint coeffs. to RHS (4)
            #   Minimum and maximum ratios across positive and negative right-hand-sides (RHS)
            prhs_ratio_max, prhs_ratio_min = -1, 1
            nrhs_ratio_max, nrhs_ratio_min = -1, 1
            for neighbor_index in range(nb_neighbors):
                coef = nonzero_coefs_raw[neighbor_index]
                rhs = SCIProwGetRhs(neighbors[neighbor_index])
                lhs = SCIProwGetLhs(neighbors[neighbor_index])
                if not SCIPisInfinity(scip, REALABS(rhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(rhs))
                    if rhs >= 0:
                        rhs_ratio_max = max(prhs_ratio_max, value)
                        rhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)
                if not SCIPisInfinity(scip, REALABS(lhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(lhs))
                    if -lhs >= 0:
                        prhs_ratio_max = max(prhs_ratio_max, value)
                        prhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)
            cand_prhs_ratio_max[cand_i] = prhs_ratio_max
            cand_prhs_ratio_min[cand_i] = prhs_ratio_min
            cand_nrhs_ratio_max[cand_i] = nrhs_ratio_max
            cand_nrhs_ratio_min[cand_i] = nrhs_ratio_min

            # Min/max for one-to-all coefficient ratios (8)
            #   The statistics are over the ratios of a variable’s coefficient, to the sum over all
            #   other variables’ coefficients, for a given constraint. Four versions of these ratios
            #   are considered: positive (negative) coefficient to sum of positive (negative) coefficients
            ota_pp_max, ota_pp_min, ota_pn_max, ota_pn_min = 0, 1, 0, 1
            ota_np_max, ota_np_min, ota_nn_max, ota_nn_min = 0, 1, 0, 1
            for neighbor_index in range(nb_neighbors):
                all_coefs_raw = SCIProwGetVals(neighbors[neighbor_index])
                neighbor_ncolumns = SCIProwGetNNonz(neighbors[neighbor_index])
                pos_coef_sum, neg_coef_sum = 0, 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    neighbor_coef = all_coefs_raw[neighbor_column_index]
                    if neighbor_coef > 0:
                        pos_coef_sum += neighbor_coef
                    else:
                        neg_coef_sum += neighbor_coef
                coef = nonzero_coefs_raw[neighbor_index]
                if coef > 0:
                    pratio = coef / pos_coef_sum
                    nratio = coef / (coef - neg_coef_sum)
                    ota_pp_max = max(ota_pp_max, pratio)
                    ota_pp_min = min(ota_pp_min, pratio)
                    ota_pn_max = max(ota_pn_max, nratio)
                    ota_pn_min = min(ota_pn_min, nratio)
                if coef < 0:
                    pratio = coef / (coef - pos_coef_sum)
                    nratio = coef / neg_coef_sum
                    ota_np_max = max(ota_np_max, pratio)
                    ota_np_min = min(ota_np_min, pratio)
                    ota_nn_max = max(ota_nn_max, nratio)
                    ota_nn_min = min(ota_nn_min, nratio)
            cand_ota_pp_max[cand_i] = ota_pp_max
            cand_ota_pp_min[cand_i] = ota_pp_min
            cand_ota_pn_max[cand_i] = ota_pn_max
            cand_ota_pn_min[cand_i] = ota_pn_min
            cand_ota_np_max[cand_i] = ota_np_max
            cand_ota_np_min[cand_i] = ota_np_min
            cand_ota_nn_max[cand_i] = ota_nn_max
            cand_ota_nn_min[cand_i] = ota_nn_min

        # Active dynamic
        # --------------
        # Stats. for active constraint coefficients (24)
        #   An active constraint at a node LP is one which is binding with equality at the optimum.
        #   We consider 4 weighting schemes for an active constraint: unit weight, inverse of the
        #   sum of the coefficients of all variables in constraint, inverse of the sum of the coefficients
        #   of only candidate variables in constraint, dual cost of the constraint. Given the absolute
        #   value of the coefficients of xj in the active constraints, we compute the sum, mean, stdev.,
        #   max. and min. of those values, for each of the weighting schemes. We also compute the weighted
        #   number of active constraints that xj is in, with the same 4 weightings
        cdef int row_index
        cdef int nrows = SCIPgetNLPRows(scip)
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef float constraint_sum, abs_coef
        cdef SCIP_COL** neighbor_columns
        cdef int neighbor_var_index, candidate_index
        cdef int active_count
        cdef float acons_sum1, acons_mean1, acons_var1, acons_max1, acons_min1
        cdef float acons_sum2, acons_mean2, acons_var2, acons_max2, acons_min2
        cdef float acons_sum3, acons_mean3, acons_var3, acons_max3, acons_min3
        cdef float acons_sum4, acons_mean4, acons_var4, acons_max4, acons_min4
        cdef float acons_nb1, acons_nb2, acons_nb3, acons_nb4
        cdef np.ndarray[np.float32_t, ndim=1] act_cons_w1, act_cons_w2, act_cons_w3, act_cons_w4

        act_cons_w1 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w2 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w3 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w4 = np.zeros(shape=(nrows, ), dtype=np.float32)
        for row_index in range(nrows):
            row = rows[row_index]
            rhs = SCIProwGetRhs(row)
            lhs = SCIProwGetLhs(row)
            activity = SCIPgetRowActivity(scip, row)
            # N.B. active if activity = lhs or rhs
            if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                neighbor_columns = SCIProwGetCols(row)
                neighbor_ncolumns = SCIProwGetNNonz(row)
                neighbor_columns_values = SCIProwGetVals(row)

                # weight no. 1
                # unit weight
                act_cons_w1[row_index] = 1

                # weight no. 2
                # inverse of the sum of the coefficients of all variables in constraint
                constraint_sum = 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    constraint_sum += REALABS(neighbor_columns_values[neighbor_column_index])
                act_cons_w2[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum

                # weight no. 3
                # inverse of the sum of the coefficients of only candidate variables in constraint
                constraint_sum = 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    neighbor_var = SCIPcolGetVar(neighbor_columns[neighbor_column_index])
                    neighbor_var_index = SCIPvarGetIndex(neighbor_var)
                    for cand_i in range(ncands):
                        var = (<Variable>candidates[cand_i]).scip_var
                        if SCIPvarGetIndex(var) == neighbor_var_index:
                            constraint_sum += REALABS(neighbor_columns_values[neighbor_column_index])
                            break
                act_cons_w3[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum

                # weight no. 4
                # dual cost of the constraint
                act_cons_w4[row_index] = REALABS(SCIProwGetDualsol(row))

        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)

            acons_sum1, acons_mean1, acons_var1, acons_max1, acons_min1 = 0, 0, 0, 0, 0
            acons_sum2, acons_mean2, acons_var2, acons_max2, acons_min2 = 0, 0, 0, 0, 0
            acons_sum3, acons_mean3, acons_var3, acons_max3, acons_min3 = 0, 0, 0, 0, 0
            acons_sum4, acons_mean4, acons_var4, acons_max4, acons_min4 = 0, 0, 0, 0, 0
            acons_nb1,  acons_nb2,   acons_nb3,   acons_nb4             = 0, 0, 0, 0
            active_count = 0
            for neighbor_index in range(nb_neighbors):
                rhs = SCIProwGetRhs(neighbors[neighbor_index])
                lhs = SCIProwGetLhs(neighbors[neighbor_index])
                activity = SCIPgetRowActivity(scip, neighbors[neighbor_index])
                # N.B. active if activity = lhs or rhs
                if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                    active_count += 1
                    neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                    abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                    acons_nb1 += act_cons_w1[neighbor_row_index]
                    value = act_cons_w1[neighbor_row_index] * abs_coef
                    acons_sum1 += value
                    acons_max1 = value if active_count == 1 else max(acons_max1, value)
                    acons_min1 = value if active_count == 1 else min(acons_min1, value)

                    acons_nb2 += act_cons_w2[neighbor_row_index]
                    value = act_cons_w2[neighbor_row_index] * abs_coef
                    acons_sum2 += value
                    acons_max2 = value if active_count == 1 else max(acons_max2, value)
                    acons_min2 = value if active_count == 1 else min(acons_min2, value)

                    acons_nb3 += act_cons_w3[neighbor_row_index]
                    value = act_cons_w3[neighbor_row_index] * abs_coef
                    acons_sum3 += value
                    acons_max3 = value if active_count == 1 else max(acons_max3, value)
                    acons_min3 = value if active_count == 1 else min(acons_min3, value)

                    acons_nb4 += act_cons_w4[neighbor_row_index]
                    value = act_cons_w4[neighbor_row_index] * abs_coef
                    acons_sum4 += value
                    acons_max4 = value if active_count == 1 else max(acons_max4, value)
                    acons_min4 = value if active_count == 1 else min(acons_min4, value)

            if active_count > 0:
                acons_mean1 = acons_sum1 / active_count
                acons_mean2 = acons_sum2 / active_count
                acons_mean3 = acons_sum3 / active_count
                acons_mean4 = acons_sum4 / active_count
                for neighbor_index in range(nb_neighbors):
                    rhs = SCIProwGetRhs(neighbors[neighbor_index])
                    lhs = SCIProwGetLhs(neighbors[neighbor_index])
                    activity = SCIPgetRowActivity(scip, neighbors[neighbor_index])
                    # N.B. active if activity = lhs or rhs
                    if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                        neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                        abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                        value = act_cons_w1[neighbor_row_index] * abs_coef
                        acons_var1 += (value - acons_mean1)**2

                        value = act_cons_w2[neighbor_row_index] * abs_coef
                        acons_var2 += (value - acons_mean2)**2

                        value = act_cons_w3[neighbor_row_index] * abs_coef
                        acons_var3 += (value - acons_mean3)**2

                        value = act_cons_w4[neighbor_row_index] * abs_coef
                        acons_var4 += (value - acons_mean4)**2
                acons_var1 /= active_count
                acons_var2 /= active_count
                acons_var3 /= active_count
                acons_var4 /= active_count

            cand_acons_sum1[cand_i]  = acons_sum1
            cand_acons_sum2[cand_i]  = acons_sum2
            cand_acons_sum3[cand_i]  = acons_sum3
            cand_acons_sum4[cand_i]  = acons_sum4
            cand_acons_mean1[cand_i] = acons_mean1
            cand_acons_mean2[cand_i] = acons_mean2
            cand_acons_mean3[cand_i] = acons_mean3
            cand_acons_mean4[cand_i] = acons_mean4
            cand_acons_max1[cand_i]  = acons_max1
            cand_acons_max2[cand_i]  = acons_max2
            cand_acons_max3[cand_i]  = acons_max3
            cand_acons_max4[cand_i]  = acons_max4
            cand_acons_min1[cand_i]  = acons_min1
            cand_acons_min2[cand_i]  = acons_min2
            cand_acons_min3[cand_i]  = acons_min3
            cand_acons_min4[cand_i]  = acons_min4
            cand_acons_var1[cand_i]  = acons_var1
            cand_acons_var2[cand_i]  = acons_var2
            cand_acons_var3[cand_i]  = acons_var3
            cand_acons_var4[cand_i]  = acons_var4
            cand_acons_nb1[cand_i]   = acons_nb1
            cand_acons_nb2[cand_i]   = acons_nb2
            cand_acons_nb3[cand_i]   = acons_nb3
            cand_acons_nb4[cand_i]   = acons_nb4

        return {
            'coefs':                cand_coefs,
            'coefs_pos':            cand_coefs_pos,
            'coefs_neg':            cand_coefs_neg,
            'nnzrs':                cand_nnzrs,
            'root_cdeg_mean':       cand_root_cdeg_mean,
            'root_cdeg_var':        cand_root_cdeg_var,
            'root_cdeg_min':        cand_root_cdeg_min,
            'root_cdeg_max':        cand_root_cdeg_max,
            'root_pcoefs_count':    cand_root_pcoefs_count,
            'root_pcoefs_mean':     cand_root_pcoefs_mean,
            'root_pcoefs_var':      cand_root_pcoefs_var,
            'root_pcoefs_min':      cand_root_pcoefs_min,
            'root_pcoefs_max':      cand_root_pcoefs_max,
            'root_ncoefs_count':    cand_root_ncoefs_count,
            'root_ncoefs_mean':     cand_root_ncoefs_mean,
            'root_ncoefs_var':      cand_root_ncoefs_var,
            'root_ncoefs_min':      cand_root_ncoefs_min,
            'root_ncoefs_max':      cand_root_ncoefs_max,
            'solfracs':             cand_solfracs,
            'slack':                cand_slack,
            'ps_up':                cand_ps_up,
            'ps_down':              cand_ps_down,
            'ps_ratio':             cand_ps_ratio,
            'ps_sum':               cand_ps_sum,
            'ps_product':           cand_ps_product,
            'nb_up_infeas':         cand_nb_up_infeas,
            'nb_down_infeas':       cand_nb_down_infeas,
            'frac_up_infeas':       cand_frac_up_infeas,
            'frac_down_infeas':     cand_frac_down_infeas,
            'cdeg_mean':            cand_cdeg_mean,
            'cdeg_var':             cand_cdeg_var,
            'cdeg_min':             cand_cdeg_min,
            'cdeg_max':             cand_cdeg_max,
            'cdeg_mean_ratio':      cand_cdeg_mean_ratio,
            'cdeg_min_ratio':       cand_cdeg_min_ratio,
            'cdeg_max_ratio':       cand_cdeg_max_ratio,
            'prhs_ratio_max':       cand_prhs_ratio_max,
            'prhs_ratio_min':       cand_prhs_ratio_min,
            'nrhs_ratio_max':       cand_nrhs_ratio_max,
            'nrhs_ratio_min':       cand_nrhs_ratio_min,
            'ota_pp_max':           cand_ota_pp_max,
            'ota_pp_min':           cand_ota_pp_min,
            'ota_pn_max':           cand_ota_pn_max,
            'ota_pn_min':           cand_ota_pn_min,
            'ota_np_max':           cand_ota_np_max,
            'ota_np_min':           cand_ota_np_min,
            'ota_nn_max':           cand_ota_nn_max,
            'ota_nn_min':           cand_ota_nn_min,
            'acons_sum1':           cand_acons_sum1,
            'acons_mean1':          cand_acons_mean1,
            'acons_var1':           cand_acons_var1,
            'acons_max1':           cand_acons_max1,
            'acons_min1':           cand_acons_min1,
            'acons_sum2':           cand_acons_sum2,
            'acons_mean2':          cand_acons_mean2,
            'acons_var2':           cand_acons_var2,
            'acons_max2':           cand_acons_max2,
            'acons_min2':           cand_acons_min2,
            'acons_sum3':           cand_acons_sum3,
            'acons_mean3':          cand_acons_mean3,
            'acons_var3':           cand_acons_var3,
            'acons_max3':           cand_acons_max3,
            'acons_min3':           cand_acons_min3,
            'acons_sum4':           cand_acons_sum4,
            'acons_mean4':          cand_acons_mean4,
            'acons_var4':           cand_acons_var4,
            'acons_max4':           cand_acons_max4,
            'acons_min4':           cand_acons_min4,
            'acons_nb1':            cand_acons_nb1,
            'acons_nb2':            cand_acons_nb2,
            'acons_nb3':            cand_acons_nb3,
            'acons_nb4':            cand_acons_nb4,
        }

    def getSolvingStats(self):
        cdef SCIP* scip = self._scip

        # recover open nodes
        cdef SCIP_NODE **leaves
        cdef SCIP_NODE **children
        cdef SCIP_NODE **siblings
        cdef int nleaves, nchildren, nsiblings, i

        PY_SCIP_CALL(SCIPgetOpenNodesData(scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings))

        # recover upper and lower bounds
        cdef np.float_t primalbound = SCIPgetPrimalbound(scip)
        cdef np.float_t dualbound = SCIPgetDualbound(scip)

        # record open node quantiles
        cdef np.ndarray[np.float_t, ndim=1] lowerbounds = np.empty([nleaves+nchildren+nsiblings+1], dtype=np.float)

        lowerbounds[0] = dualbound
        for i in range(nleaves):
            lowerbounds[1+i] = leaves[i].lowerbound
        for i in range(nchildren):
            lowerbounds[1+nleaves+i] = children[i].lowerbound
        for i in range(nsiblings):
            lowerbounds[1+nleaves+nchildren+i] = siblings[i].lowerbound

        percentiles = (10, 25, 50, 75, 90)
        qs = np.percentile(lowerbounds, percentiles, overwrite_input=True, interpolation='linear')

        return {

            # open nodes (parent's) dual bounds quantiles
            'opennodes_10quant': qs[0],
            'opennodes_25quant': qs[1],
            'opennodes_50quant': qs[2],
            'opennodes_75quant': qs[3],
            'opennodes_90quant': qs[4],

            # hardcoded statistics
            'ninternalnodes': scip.stat.ninternalnodes,
            'ncreatednodes': scip.stat.ncreatednodes,
            'ncreatednodesrun': scip.stat.ncreatednodesrun,
            'nactivatednodes': scip.stat.nactivatednodes,
            'ndeactivatednodes': scip.stat.ndeactivatednodes,

            # http://scip.zib.de/doc/html/group__PublicLPMethods.php

            'lp_obj': SCIPgetLPObjval(scip),

            # http://scip.zib.de/doc/html/group__PublicTreeMethods.php

            'depth': SCIPgetDepth(scip),
            'focusdepth': SCIPgetFocusDepth(scip),
            'plungedepth': SCIPgetPlungeDepth(scip),
            'effectiverootdepth': SCIPgetEffectiveRootDepth(scip),
            'inrepropagation': SCIPinRepropagation(scip),
            'nleaves': SCIPgetNLeaves(scip),
            'nnodesleft': SCIPgetNNodesLeft(scip),  # nleaves + nchildren + nsiblings
            'cutoffdepth': SCIPgetCutoffdepth(scip),  # depth of first node in active path that is marked being cutoff
            'repropdepth': SCIPgetRepropdepth(scip),  # depth of first node in active path that has to be propagated again

            # http://scip.zib.de/doc/html/group__PublicSolvingStatsMethods.php

            'nruns': SCIPgetNRuns(scip),
            'nreoptruns': SCIPgetNReoptRuns(scip),
            'nnodes': SCIPgetNNodes(scip),
            'ntotalnodes': SCIPgetNTotalNodes(scip),  # total number of processed nodes in all runs, including the focus node
            'nfeasibleleaves': SCIPgetNFeasibleLeaves(scip),  # number of leaf nodes processed with feasible relaxation solution
            'ninfeasibleleaves': SCIPgetNInfeasibleLeaves(scip),  # number of infeasible leaf nodes processed
            'nobjlimleaves': SCIPgetNObjlimLeaves(scip),  # number of processed leaf nodes that hit LP objective limit
            'ndelayedcutoffs': SCIPgetNDelayedCutoffs(scip),  # gets number of times a selected node was from a cut off subtree
            'nlps': SCIPgetNLPs(scip),
            'nlpiterations': SCIPgetNLPIterations(scip),
            'nnzs': SCIPgetNNZs(scip),
            'nrootlpiterations': SCIPgetNRootLPIterations(scip),
            'nrootfirstlpiterations': SCIPgetNRootFirstLPIterations(scip),
            'nprimallps': SCIPgetNPrimalLPs(scip),
            'nprimallpiterations': SCIPgetNPrimalLPIterations(scip),
            'nduallps': SCIPgetNDualLPs(scip),
            'nduallpiterations': SCIPgetNDualLPIterations(scip),
            'nbarrierlps': SCIPgetNBarrierLPs(scip),
            'nbarrierlpiterations': SCIPgetNBarrierLPIterations(scip),
            'nresolvelps': SCIPgetNResolveLPs(scip),
            'nresolvelpiterations': SCIPgetNResolveLPIterations(scip),
            'nprimalresolvelps': SCIPgetNPrimalResolveLPs(scip),
            'nprimalresolvelpiterations': SCIPgetNPrimalResolveLPIterations(scip),
            'ndualresolvelps': SCIPgetNDualResolveLPs(scip),
            'ndualresolvelpiterations': SCIPgetNDualResolveLPIterations(scip),
            'nnodelps': SCIPgetNNodeLPs(scip),
            'nnodelpiterations': SCIPgetNNodeLPIterations(scip),
            'nnodeinitlps': SCIPgetNNodeInitLPs(scip),
            'nnodeinitlpiterations': SCIPgetNNodeInitLPIterations(scip),
            'ndivinglps': SCIPgetNDivingLPs(scip),
            'ndivinglpiterations': SCIPgetNDivingLPIterations(scip),
            'nstrongbranchs': SCIPgetNStrongbranchs(scip),
            'nstrongbranchlpiterations': SCIPgetNStrongbranchLPIterations(scip),
            'nrootstrongbranchs': SCIPgetNRootStrongbranchs(scip),
            'nrootstrongbranchlpiterations': SCIPgetNRootStrongbranchLPIterations(scip),
            'npricerounds': SCIPgetNPriceRounds(scip),
            'npricevars': SCIPgetNPricevars(scip),
            'npricevarsfound': SCIPgetNPricevarsFound(scip),
            'npricevarsapplied': SCIPgetNPricevarsApplied(scip),
            'nseparounds': SCIPgetNSepaRounds(scip),
            'ncutsfound': SCIPgetNCutsFound(scip),
            'ncutsfoundround': SCIPgetNCutsFoundRound(scip),
            'ncutsapplied': SCIPgetNCutsApplied(scip),
            'nconflictconssfound': SCIPgetNConflictConssFound(scip),
            'nconflictconssfoundnode': SCIPgetNConflictConssFoundNode(scip),
            'nconflictconssapplied': SCIPgetNConflictConssApplied(scip),
            'maxdepth': SCIPgetMaxDepth(scip),  # maximal depth of all processed nodes in current branch and bound run (excluding probing nodes)
            'maxtotaldepth': SCIPgetMaxTotalDepth(scip),
            'nbacktracks': SCIPgetNBacktracks(scip),  # total number of backtracks, i.e. number of times, the new node was selected from the leaves queue
            'nactivesonss': SCIPgetNActiveConss(scip),
            'nenabledconss': SCIPgetNEnabledConss(scip),
            'avgdualbound': SCIPgetAvgDualbound(scip),  # average dual bound of all unprocessed nodes for original problem
            'avglowerbound': SCIPgetAvgLowerbound(scip),  # average lower (dual) bound of all unprocessed nodes in transformed problem
            'dualbound': SCIPgetDualbound(scip),  # global dual bound
            'lowerbound': SCIPgetLowerbound(scip),  # global lower (dual) bound in transformed problem
            'dualboundroot': SCIPgetDualboundRoot(scip),  # gets dual bound of the root node for the original problem
            'lowerboundroot': SCIPgetLowerboundRoot(scip),  # gets lower (dual) bound in transformed problem of the root node
            'firstlpdualboundroot': SCIPgetFirstLPDualboundRoot(scip),  # gets dual bound for the original problem obtained by the first LP solve at the root node
            'firstlplowerboundroot': SCIPgetFirstLPLowerboundRoot(scip),  # gets lower (dual) bound in transformed problem obtained by the first LP solve at the root node
            'firstprimalbound': SCIPgetFirstPrimalBound(scip),  # the primal bound of the very first solution
            'primalbound': SCIPgetPrimalbound(scip),  # gets global primal bound (objective value of best solution or user objective limit) for the original problem
            'upperbound': SCIPgetUpperbound(scip),  # gets global upper (primal) bound in transformed problem (objective value of best solution or user objective limit)
            'cutoffbound': SCIPgetCutoffbound(scip),
            'isprimalboundsol': SCIPisPrimalboundSol(scip),
            'gap': SCIPgetGap(scip),
            'transgap': SCIPgetTransGap(scip),
            'nsolsfound': SCIPgetNSolsFound(scip),
            'nlimsolsfound': SCIPgetNLimSolsFound(scip),
            'nbestsolsfound': SCIPgetNBestSolsFound(scip),
            # SCIPgetAvgPseudocost(scip, SCIP_Real solvaldelta)
            # SCIPgetAvgPseudocostCurrentRun(scip, SCIP_Real solvaldelta)
            # SCIPgetAvgPseudocostCount(scip, SCIP_BRANCHDIR dir)
            # SCIPgetAvgPseudocostCountCurrentRun(scip, SCIP_BRANCHDIR dir)
            # SCIPgetPseudocostCount(scip, SCIP_BRANCHDIR dir, SCIP_Bool onlycurrentrun)
            'avgpseudocostscore': SCIPgetAvgPseudocostScore(scip),
            # SCIPgetPseudocostVariance(scip, SCIP_BRANCHDIR branchdir, SCIP_Bool onlycurrentrun)
            'avgpseudocostscorecurrentrun': SCIPgetAvgPseudocostScoreCurrentRun(scip),
            'avgconflictscore': SCIPgetAvgConflictScore(scip),
            'avgconflictscorecurrentrun': SCIPgetAvgConflictScoreCurrentRun(scip),
            'avgconfliclengthscore': SCIPgetAvgConflictlengthScore(scip),
            'avgconfliclengthscorecurrentrun': SCIPgetAvgConflictlengthScoreCurrentRun(scip),
            # SCIPgetAvgInferences(scip, SCIP_BRANCHDIR dir)
            # SCIPgetAvgInferencesCurrentRun(scip, SCIP_BRANCHDIR dir)
            'avginferencescore': SCIPgetAvgInferenceScore(scip),
            'avginferencescorecurrentrun': SCIPgetAvgInferenceScoreCurrentRun(scip),
            # SCIPgetAvgCutoffs(scip, SCIP_BRANCHDIR dir)
            # SCIPgetAvgCutoffsCurrentRun(scip, SCIP_BRANCHDIR dir)
            'avgcutoffscore': SCIPgetAvgCutoffScore(scip),
            'avgcuroffscorecurrentrun': SCIPgetAvgCutoffScoreCurrentRun(scip),
            'deterministictime': SCIPgetDeterministicTime(scip),  # gets deterministic time number of LPs solved so far
            'solvingtime': SCIPgetSolvingTime(scip),
        }

    def executeBranchRule(self, str name, allowaddcons):
        cdef SCIP_BRANCHRULE*  branchrule
        cdef SCIP_RESULT result
        branchrule = SCIPfindBranchrule(self._scip, name.encode("UTF-8"))
        if branchrule == NULL:
            print("Error, branching rule not found!")
            return PY_SCIP_RESULT.DIDNOTFIND
        else:
            branchrule.branchexeclp(self._scip, branchrule, allowaddcons, &result)
            return result

    def getConsVals(self):
        cdef SCIP* scip = self._scip    
        cdef SCIP_CONS** conss = SCIPgetConss(scip)
        cdef int ncons = SCIPgetNConss(scip)
        # SCIPgetOrigVars
        cdef int nvars = SCIPgetNOrigVars(scip)

        cdef np.ndarray[np.float32_t, ndim=1] rhs_vec
        cdef np.ndarray[np.float32_t, ndim=1] lhs_vec
        cdef np.ndarray[np.float32_t, ndim=2] cons_matrix

        lhs_vec = np.empty(shape=(ncons, ), dtype=np.float32)
        rhs_vec = np.empty(shape=(ncons, ), dtype=np.float32)
        cons_matrix = np.zeros(shape=(ncons, nvars), dtype=np.float32)

        cdef SCIP_VAR** vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_Real* vals = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        cdef int nconsvars = 0
        cdef SCIP_Bool success = 0
        cdef int var_idx = 0
        for i in range(ncons):
            
            # fill up rhs vector
            rhs_vec[i] = SCIPconsGetRhs(scip, conss[i], &success)
            lhs_vec[i] = SCIPconsGetLhs(scip, conss[i], &success)

            # fill up cons matrix
            SCIPgetConsNVars(scip, conss[i], &nconsvars, &success)
            SCIPgetConsVars(scip, conss[i], vars, nconsvars, &success)
            SCIPgetConsVals(scip, conss[i], vals, nconsvars, &success)
            
            for j in range(nconsvars):
                varname = bytes(SCIPvarGetName(vars[j])).decode('utf-8')
                var_idx = int(varname[1:]) - 1
                # printf("%d\n", var_idx)
                cons_matrix[i][var_idx] = vals[j]

        free(<void *>vals)
        free(<void *>vars)
        return {
            "cons_matrix": cons_matrix,
            "rhs_vec": rhs_vec,
            "lhs_vec": lhs_vec
        }

# debugging memory management
def is_memory_freed():
    return BMSgetMemoryUsed() == 0

def print_memory_in_use():
    BMScheckEmptyMemory()


