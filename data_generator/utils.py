import numpy as np
import pyscipopt as scip

def init_scip_params(model, seed, heuristics=False, presolving=False, separating=False, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', False)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def extract_ding_variable_features(model):
    """
    Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """

    col_state = model.getDingStateCols()
    col_feature_names = sorted(col_state)
    for index, name in enumerate(col_feature_names):
        if name == 'col_coefs':
            break

    col_state = np.stack([col_state[feature_name] for feature_name in col_feature_names], axis=1)

    row_state = model.getDingStateRows()
    row_feature_names = sorted(row_state)
    row_state = np.stack([row_state[feature_name] for feature_name in row_feature_names], axis=1)

    vc, vo, co = model.getDingStateLPgraph()

    return (col_state, row_state, vc, vo, co), index