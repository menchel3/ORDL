import numpy as np
import os

from cfr_function.logger import Logger as Log
from cfr_function.loader import *

POL_CURVE_RES = 40

class NaNException(Exception):
    pass

def AUC_func(label, pred):
    """
    Calculate Area Under the Curve (AUC) for binary classification.
    """
    # Check if labels are binary (0 and 1)
    unique_labels = np.unique(label)
    if len(unique_labels) > 2 or (len(unique_labels) == 2 and not np.all(np.isin(unique_labels, [0, 1]))):
        # If not binary 0/1, return NaN (e.g. for continuous outcomes like IHDP)
        return np.nan
        
    if len(unique_labels) < 2:
        # Only one class present
        return np.nan

    # Sort by prediction in descending order
    desc_score_indices = np.argsort(pred)[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]

    # Calculate TPR and FPR
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Calculate AUC using trapezoidal rule
    # Insert 0 at the beginning to start from (0,0)
    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)
    
    auc = np.trapz(tpr, fpr)
    return auc

def policy_range(n, res=10):
    step = int(float(n)/float(res))
    if step == 0:
        step = 1
    n_range = list(range(0,int(n+1),step))
    if not n_range[-1] == n:
        n_range.append(n)

    # To make sure every curve is same length. Incurs a small error if res high.
    # Only occurs if number of units considered differs.
    # For example if resampling validation sets (with different number of
    # units in the randomized sub-population)

    while len(n_range) > res:
        k = np.random.randint(len(n_range)-2)+1
        del n_range[k]

    return n_range

def policy_val(t, yf, eff_pred, compute_policy_curve=False):
    """ Computes the value of the policy defined by predicted effect """

    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    treat_overlap = (policy==t)*(t>0)
    control_overlap = (policy==t)*(t<1)

    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    policy_curve = []

    if compute_policy_curve:
        n = t.shape[0]
        I_sort = np.argsort(-eff_pred)

        n_range = policy_range(n, POL_CURVE_RES)

        for i in n_range:
            I = I_sort[0:i]

            policy_i = 0*policy
            policy_i[I] = 1
            pit_i = np.mean(policy_i)

            treat_overlap = (policy_i>0)*(t>0)
            control_overlap = (policy_i<1)*(t<1)

            if np.sum(treat_overlap)==0:
                treat_value = 0
            else:
                treat_value = np.mean(yf[treat_overlap])

            if np.sum(control_overlap)==0:
                control_value = 0
            else:
                control_value = np.mean(yf[control_overlap])

            policy_curve.append(pit_i*treat_value + (1-pit_i)*control_value)

    return policy_value, policy_curve

def pdist2(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)

def cf_nn(x, t):
    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    x_c = x[Ic,:]
    x_t = x[It,:]

    D = pdist2(x_c, x_t)

    nn_t = Ic[np.argmin(D,0)]
    nn_c = It[np.argmin(D,1)]

    return nn_t, nn_c

def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None, mode='ATE'):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x,t)

    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    if mode=='ATE':
        ycf_t = 1.0*y[nn_t]
        eff_nn_t = ycf_t - 1.0*y[It]
        eff_pred_t = ycf_p[It] - yf_p[It]

        eff_pred = eff_pred_t
        eff_nn = eff_nn_t

        ycf_c = 1.0*y[nn_c]
        eff_nn_c = ycf_c - 1.0*y[Ic]
        eff_pred_c = ycf_p[Ic] - yf_p[Ic]

        eff_pred = np.concatenate((eff_pred_t, eff_pred_c),axis=0)
        eff_nn = np.concatenate((eff_nn_t, eff_nn_c),axis=0)
    elif mode=='ATT':
        ycf_t = 1.0*y[nn_c]
        eff_nn_t = ycf_t - 1.0*y[Ic]
        eff_pred_t = ycf_p - yf_p

        eff_pred = eff_pred_t
        eff_nn = eff_nn_t
    
    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))

    return pehe_nn

def evaluate_bin_att(predictions, data, i_exp, I_subset=None,
                     compute_policy_curve=False, nn_t=None, nn_c=None, bin_or_cont = 1):

    x = data['x'][:,:,i_exp]
    t = data['t'][:,i_exp]
    e = data['e'][:,i_exp]
    yf = data['yf'][:,i_exp]
    yf_p = predictions[:,0]
    ycf_p = predictions[:,1]
    if bin_or_cont == 0:
        yf_p = 1.0 * (yf_p > 0.5)
        ycf_p = 1.0 * (ycf_p > 0.5)

    att = np.mean(yf[t>0]) - np.mean(yf[(1-t+e)>1])

    if not I_subset is None:
        x = x[I_subset,:]
        t = t[I_subset]
        e = e[I_subset]
        yf_p = yf_p[I_subset]
        ycf_p = ycf_p[I_subset]
        yf = yf[I_subset]

    yf_p_b = 1.0*(yf_p>0.5)
    ycf_p_b = 1.0*(ycf_p>0.5)

    if np.any(np.isnan(yf_p)) or np.any(np.isnan(ycf_p)):
        raise NaNException('NaN encountered')

    #IMPORTANT: NOT USING BINARIZATION FOR EFFECT, ONLY FOR CLASSIFICATION!

    eff_pred = ycf_p - yf_p;
    eff_pred[t>0] = -eff_pred[t>0];

    ate_pred = np.mean(eff_pred[e>0])
    atc_pred = np.mean(eff_pred[(1-t+e)>1])

    att_pred = np.mean(eff_pred[(t+e)>1])
    bias_att = np.abs(att_pred - att)

    err_fact = np.mean(np.abs(yf_p_b-yf))

    p1t = np.mean(yf[t>0])
    p1t_p = np.mean(yf_p[t>0])

    lpr = np.log(p1t / p1t_p + 0.001)

    policy_value, policy_curve = \
        policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'bias_att': bias_att, 'atc_pred': atc_pred,
            'err_fact': err_fact, 'lpr': lpr,
            'policy_value': policy_value, 'policy_risk': 1-policy_value,
            'policy_curve': policy_curve, 'pehe_nn': pehe_appr}

def evaluate_cont_ate(predictions, data, i_exp, I_subset=None,
    compute_policy_curve=False, nn_t=None, nn_c=None, bin_or_cont = 1):

    x = data['x'][:,:,i_exp]
    t = data['t'][:,i_exp]
    yf = data['yf'][:,i_exp]
    
    # Check if dataset has mu0 and mu1 (IHDP) or not (TWINS)
    has_mu = 'mu0' in data and 'mu1' in data and data['mu0'] is not None and data['mu1'] is not None
    has_ycf = 'ycf' in data and data['ycf'] is not None
    
    if has_ycf:
        ycf = data['ycf'][:,i_exp]
    
    if has_mu:
        mu0 = data['mu0'][:,i_exp]
        mu1 = data['mu1'][:,i_exp]
    
    yf_p = predictions[:,0]
    ycf_p = predictions[:,1]
    if bin_or_cont == 0:
        yf_p = 1.0 * (yf_p > 0.5)
        ycf_p = 1.0 * (ycf_p > 0.5)

    if not I_subset is None:
        x = x[I_subset,]
        t = t[I_subset]
        yf_p = yf_p[I_subset]
        ycf_p = ycf_p[I_subset]
        yf = yf[I_subset]
        if has_ycf:
            ycf = ycf[I_subset]
        if has_mu:
            mu0 = mu0[I_subset]
            mu1 = mu1[I_subset]

    # Calculate ground truth effect
    # For IHDP: use mu1 - mu0 (always treatment effect)
    # For TWINS: ITE = Y(1) - Y(0), so:
    #   - For controls (t=0): ITE = ycf - yf (counterfactual is Y(1), factual is Y(0))
    #   - For treated (t=1): ITE = yf - ycf (factual is Y(1), counterfactual is Y(0))
    if has_mu:
        eff = mu1 - mu0
    elif has_ycf:
        # TWINS: calculate ITE correctly based on treatment assignment
        eff = np.zeros_like(yf)
        eff[t < 1] = ycf[t < 1] - yf[t < 1]  # Controls: Y(1) - Y(0)
        eff[t > 0] = yf[t > 0] - ycf[t > 0]  # Treated: Y(1) - Y(0)
    else:
        # No ground truth available, use predictions as placeholder
        eff = np.zeros_like(yf_p)
        eff[t < 1] = ycf_p[t < 1] - yf_p[t < 1]
        eff[t > 0] = yf_p[t > 0] - ycf_p[t > 0]

    # Calculate predicted effect
    # ycf_p is prediction of counterfactual, yf_p is prediction of factual
    # For controls (t=0): ITE = ycf_p - yf_p (predicted Y(1) - predicted Y(0))
    # For treated (t=1): ITE = yf_p - ycf_p (predicted Y(1) - predicted Y(0))
    eff_pred = np.zeros_like(yf_p)
    eff_pred[t < 1] = ycf_p[t < 1] - yf_p[t < 1]
    eff_pred[t > 0] = yf_p[t > 0] - ycf_p[t > 0]

    pehe = np.sqrt(np.mean(np.square(eff_pred-eff)))
    
    rmse_fact = np.sqrt(np.mean(np.square(yf_p-yf)))
    if has_ycf:
        rmse_cfact = np.sqrt(np.mean(np.square(ycf_p-ycf)))
    else:
        rmse_cfact = 0.0  # No counterfactual ground truth available
    
    # ITE prediction using predicted counterfactual and observed factual
    # For controls (t=0): ITE = ycf_p - yf (predicted Y(1) - observed Y(0))
    # For treated (t=1): ITE = yf - ycf_p (observed Y(1) - predicted Y(0))
    ite_pred = np.zeros_like(yf)
    ite_pred[t < 1] = ycf_p[t < 1] - yf[t < 1]
    ite_pred[t > 0] = yf[t > 0] - ycf_p[t > 0]
    rmse_ite = np.sqrt(np.mean(np.square(ite_pred-eff)))

    ate_pred = np.mean(eff_pred)
    ate = np.mean(eff)
    bias_ate = np.abs(ate_pred-ate)

    att_pred = np.mean(eff_pred[t>0])
    att = np.mean(eff[t>0])
    bias_att = np.abs(att_pred - att)

    atc_pred = np.mean(eff_pred[t<1])
    atc = np.mean(eff[t<1])
    bias_atc = np.abs(atc_pred - atc)

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    # Calculate AUC for factual outcome prediction
    auc = AUC_func(yf, yf_p)

    # @TODO: Not clear what this is for continuous data
    #policy_value, policy_curve = policy_val(t, yf, eff_pred, compute_policy_curve)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'atc_pred': atc_pred, 'bias_ate': bias_ate,
            'bias_att': bias_att, 'bias_atc': bias_atc,
            'rmse_fact': rmse_fact, 'rmse_cfact': rmse_cfact,
            'pehe': pehe, 'rmse_ite': rmse_ite, 'pehe_nn': pehe_appr,
            'ate':ate, 'att':att, 'atc':atc, 'auc': auc }
            #'policy_value': policy_value, 'policy_curve': policy_curve}

def evaluate_cont_att(predictionsg, data, i_exp, I_subset=None,
    compute_policy_curve=False, nn_t=None, nn_c=None, mode='ATE', bin_or_cont = 1):

    _yf = data['yf'][:,i_exp]
    x = data['x'][:,:,i_exp]
    t = data['t'][:,i_exp]
    yf = data['yf'][:,i_exp]
    ycf = data['ycf'][:,i_exp]
    mu0 = data['mu0'][:,i_exp]
    mu1 = data['mu1'][:,i_exp]
    ycf_p = predictions[:,0]
    # ycf_ = gths[:,0]
    ycf_ = data['ycf'][:,i_exp]
    if bin_or_cont == 0:
        ycf_ = 1.0 * (ycf_ > 0.5)
        ycf_p = 1.0 * (ycf_p > 0.5)

    if not I_subset is None:
        _yf = _yf[I_subset]
        x = x[I_subset,]
        t = t[I_subset]
        yf = yf[I_subset]
        ycf = ycf[I_subset]
        mu0 = mu0[I_subset]
        mu1 = mu1[I_subset]
        ycf_p = ycf_p[I_subset]
        ycf_ = ycf_[I_subset]

    if mode == 'ATT':
        index = np.where(t < 1)[0]
    elif mode == 'ATC':
        index = np.where(t > 0)[0]
    
    x = x[index,]
    t = t[index]
    yf = yf[index]
    ycf = ycf[index]
    mu0 = mu0[index]
    mu1 = mu1[index]
    ycf_p = ycf_p[index]
    ycf_ = ycf_[index]

    eff = ycf-yf
    eff[t>0] = -eff[t>0];

    rmse_cfact = np.sqrt(np.mean(np.square(ycf_p-ycf_)))
    
    eff_pred = ycf_p - yf;
    eff_pred[t>0] = -eff_pred[t>0];

    itt_pred = ycf_p - yf
    itt_pred[t>0] = -itt_pred[t>0]
    rmse_itt = np.sqrt(np.mean(np.square(itt_pred-eff)))

    att_pred = np.mean(eff_pred[t>0])
    bias_att = att_pred - np.mean(eff[t>0])

    atc_pred = np.mean(eff_pred[t<1])
    bias_atc = atc_pred - np.mean(eff[t<1])

    pehe = np.sqrt(np.mean(np.square(eff_pred-eff)))

    pehe_appr = pehe_nn(yf, ycf_p, _yf, x, t, nn_t, nn_c, mode)

    # @TODO: Not clear what this is for continuous data
    #policy_value, policy_curve = policy_val(t, yf, eff_pred, compute_policy_curve)

    if mode == 'ATT':
        ate_pred = att_pred
        bias_ate = bias_att
    elif mode == 'ATC':
        ate_pred = atc_pred
        bias_ate = bias_att

    rmse_fact = rmse_cfact
    
    return {'att_pred': att_pred, 'bias_att': bias_att, 'rmse_cfact': rmse_cfact,
            'pehe': pehe, 'rmse_itt': rmse_itt, 'pehe_nn': pehe_appr}
            #'policy_value': policy_value, 'policy_curve': policy_curve}

def evaluate_result(result, data, validation=False, multiple_exps=False, binary=False, mode='ATE', bin_or_cont = 1):

    predictions = result['pred']
    # gths = result['gth']
    gths = None


    if validation:
        I_valid = result['val']

    n_units, _, n_rep, n_outputs = predictions.shape
    # print(predictions.shape)
    # exit()

    #@TODO: Should depend on parameter
    compute_policy_curve = True

    eval_results = []
    #Loop over output_times
    for i_out in range(n_outputs):
        eval_results_out = []
        if not multiple_exps and not validation:
            nn_t, nn_c = cf_nn(data['x'][:,:,0], data['t'][:,0])


        #Loop over repeated experiments
        for i_rep in range(n_rep):
            # validation = False
            if validation:
                I_valid_rep = I_valid[i_rep,:]
            else:
                I_valid_rep = None

            if multiple_exps:
                i_exp = i_rep
                if validation:
                    nn_t, nn_c = cf_nn(data['x'][I_valid_rep,:,i_exp], data['t'][I_valid_rep,i_exp])
                else:
                    nn_t, nn_c = cf_nn(data['x'][:,:,i_exp], data['t'][:,i_exp])
            else:
                i_exp = 0

            if validation and not multiple_exps:
                nn_t, nn_c = cf_nn(data['x'][I_valid_rep,:,i_exp], data['t'][I_valid_rep,i_exp])

            if mode == 'ATE':
                if binary:
                    # Jobs dataset: use binary evaluation, calculate policy risk and ATT
                    eval_result = evaluate_bin_att(predictions[:,:,i_rep,i_out],
                        data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, bin_or_cont = bin_or_cont)
                else:
                    # IHDP/TWINS dataset: use continuous evaluation, calculate PEHE and ATE
                    eval_result = evaluate_cont_ate(predictions[:,:,i_rep,i_out],
                        data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, bin_or_cont = bin_or_cont)
            elif mode == 'ATT':
                eval_result = evaluate_cont_att(predictions[:,:,i_rep,i_out], gths, 
                        data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, mode=mode, bin_or_cont = bin_or_cont)
                # eval_result = evaluate_cont_att(predictions[:,:,i_rep,i_out], gths[:,:,i_rep,i_out], 
                #         data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, mode=mode)
            
            elif mode == 'ATC':
                eval_result = evaluate_cont_att(predictions[:,:,i_rep,i_out], gths, 
                        data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, mode=mode, bin_or_cont = bin_or_cont)
                # eval_result = evaluate_cont_att(predictions[:,:,i_rep,i_out], gths[:,:,i_rep,i_out], 
                #         data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c, mode=mode)

            eval_results_out.append(eval_result)

        eval_results.append(eval_results_out)

    # Reformat into dict
    eval_dict = {}
    keys = eval_results[0][0].keys()
    for k in keys:
        arr = [[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)]
        v = np.array([[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)])
        eval_dict[k] = v

    # Gather loss
    # Shape [times, types, reps]
    # Types: obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj
    if 'loss' in result.keys() and result['loss'].shape[1]>=6:
        losses = result['loss']
        n_loss_outputs = losses.shape[0]

        # Improved alignment logic
        if n_loss_outputs == n_outputs + 1:
             # Assume initial loss at index 0, then 1-to-1 mapping
             indices = [i + 1 for i in range(n_outputs)]
        elif n_loss_outputs == n_outputs:
             indices = [i for i in range(n_outputs)]
        else:
             # Fallback to ratio-based (existing behavior)
             indices = [int((n_loss_outputs*i)/n_outputs) for i in range(n_outputs)]

        if validation:
            objective = np.array([losses[idx,6,:] for idx in indices]).T
        else:
            objective = np.array([losses[idx,0,:] for idx in indices]).T

        eval_dict['objective'] = objective

    return eval_dict

def evaluate(output_dir, data_path_train, data_path_test=None, binary=False, mode='ATE', bin_or_cont = 1):

    print ('\nEvaluating experiment %s...' % output_dir)

    # Load results for all configurations
    results = load_results(output_dir)
    # print(results[0].keys())
    # exit()

    if len(results) == 0:
        raise Exception('No finished results found.')

    # Separate configuration files
    configs = [r['config'] for r in results]

    # Test whether multiple experiments (different data)
    multiple_exps = (configs[0]['experiments'] > 1)
    if Log.VERBOSE and multiple_exps:
        print ('Multiple data ( experiments) detected')
    # Load training data
    if Log.VERBOSE:
        print ('Loading TRAINING data %s...' % data_path_train)
    data_train = load_data(data_path_train)

    # Load test data
    if data_path_test is not None:
        if Log.VERBOSE:
            print ('Loading TEST data %s...' % data_path_test)
        data_test = load_data(data_path_test)
    else:
        data_test = None

    # Evaluate all results
    eval_results = []
    configs_out = []
    i = 0
    if Log.VERBOSE:
        print ('Evaluating result (out of %d): ' % len(results))
    for result in results:
        if Log.VERBOSE:
            print ('Evaluating %d...' % (i+1))
        try:
            eval_train = evaluate_result(result['train'], data_train,
                validation=False, multiple_exps=multiple_exps, binary=binary, mode=mode, bin_or_cont = bin_or_cont)
            #？？？？怎么会是train
            eval_valid = evaluate_result(result['train'], data_train,
                validation=True, multiple_exps=multiple_exps, binary=binary, mode=mode, bin_or_cont = bin_or_cont)

            if data_test is not None:
                eval_test = evaluate_result(result['test'], data_test,
                    validation=False, multiple_exps=multiple_exps, binary=binary, mode=mode, bin_or_cont = bin_or_cont)
            else:
                eval_test = None

            eval_results.append({'train': eval_train, 'valid': eval_valid, 'test': eval_test})
            configs_out.append(configs[i])
        except NaNException as e:
            print ('WARNING: Encountered NaN exception. Skipping.')
            print (e)

        i += 1

    # Reformat into dict
    eval_dict = {'train': {}, 'test': {}, 'valid': {}}
    keys = eval_results[0]['train'].keys()
    for k in keys:
        v = np.array([eval_results[i]['train'][k] for i in range(len(eval_results))])
        eval_dict['train'][k] = v

        v = np.array([eval_results[i]['valid'][k] for i in range(len(eval_results))])
        eval_dict['valid'][k] = v

        if eval_test is not None and k in eval_results[0]['test']:
            v = np.array([eval_results[i]['test'][k] for i in range(len(eval_results))])
            eval_dict['test'][k] = v

    return eval_dict, configs_out
