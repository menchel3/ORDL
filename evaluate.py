import sys
import os
import json
import numpy as np

import pickle

from cfr_function.logger import Logger as Log
from cfr_function.loader import load_config as load_result_config
Log.VERBOSE = True

import cfr_function.evaluation as evaluation
from cfr_function.plotting import plot_evaluation_cont, plot_evaluation_bin

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def _resolve_dataset_paths(output_dir):
    config_path = os.path.join(output_dir, 'config.txt')
    if os.path.isfile(config_path):
        cfg = load_result_config(config_path)
        datadir = cfg.get('datadir')
        dataform = cfg.get('dataform')
        data_test = cfg.get('data_test')
        if datadir and dataform and data_test:
            return os.path.join(datadir, dataform), os.path.join(datadir, data_test)

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, 'data')
    if 'jobs' in output_dir.lower():
        return (
            os.path.join(data_root, 'jobs_DW_bin.new.10.train.npz'),
            os.path.join(data_root, 'jobs_DW_bin.new.10.test.npz'),
        )
    if 'twins' in output_dir.lower():
        return (
            os.path.join(data_root, 'twins_1-10.train.npz'),
            os.path.join(data_root, 'twins_1-10.test.npz'),
        )
    return (
        os.path.join(data_root, 'ihdp_npci_1-100.train.npz'),
        os.path.join(data_root, 'ihdp_npci_1-100.test.npz'),
    )

def evaluate(output_dir='results/example_ihdp/', overwrite=True, filters=None, ground_truth=True, mode = 'ATE', bin_or_cont = 1):

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train, data_test = _resolve_dataset_paths(output_dir)

    # Auto-select dataset type based on output_dir
    if 'jobs' in output_dir.lower():
        binary = True
    elif 'twins' in output_dir.lower():
        binary = False
    else:
        binary = False

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary,mode=mode, bin_or_cont=bin_or_cont)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print ('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Call correct plotting function based on dataset type
    if binary:
        # Jobs dataset: display policy_risk, bias_att, err_fact
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        # IHDP/TWINS dataset: display pehe, bias_ate, etc.
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        with open('configs/run.json','r') as f:
            run_dict = json.load(f)
        evaluate('configs/' + run_dict['config'], overwrite = False, filters = None)
        print ('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
