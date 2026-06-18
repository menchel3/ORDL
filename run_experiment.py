"""Run the current IHDP experiment configuration."""

import subprocess
import sys
import os

IHDP_CONFIG = {
    'datadir': '/home/student1/projects/ORDL/data/',
    'dataform': 'ihdp_npci_1-100.train.npz',
    'data_test': 'ihdp_npci_1-100.test.npz',
    'outdir': 'results/ihdp/',
    'experiments':100,
    'iterations': 500,
    'batch_size': 0,
    'lrate': 1e-3,
    'lrate_decay': 0.97,
    'val_part': 0.3,
    'n_in': 7,
    'n_out': 4,
    'dim_in': 32,
    'dim_out': 256,
    'p_coef_y': 1.0,
    'p_coef_mu': 10.0,
    'p_coef_lambda': 0,
    'p_coef_mi': 0.1,
    'loss': 'l2',
    'ycf_result': 1,
    'batch_norm': 0,
    'seed': 1,
    'optimizer': 'Adam',
    'output_delay': 100,
    'pred_output_delay': 100,
}

def main():
    print("=" * 80)
    print("Running DeR-CFR IHDP Dataset Experiment")
    print("=" * 80)
    print(f"Python Interpreter: {sys.executable}")
    print(f"Dataset: {IHDP_CONFIG['dataform']}")
    print(f"Experiments: {IHDP_CONFIG['experiments']}")
    print(f"Iterations: {IHDP_CONFIG['iterations']}")
    print("=" * 80)

    cmd = [sys.executable, 'main.py']
    for key, value in IHDP_CONFIG.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}={value}')

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n[SUCCESS] Experiment completed!")
        print(f"Results location: {IHDP_CONFIG['outdir']}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed! Return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())