"""Run the current Twins experiment configuration."""

import subprocess
import sys
import os

TWINS_CONFIG = {
    'datadir': '/home/student1/projects/ORDL/data/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/twins/',
    'experiments': 10,
    'iterations': 500,
    'batch_size': 0,
    'lrate': 1e-3,
    'lrate_decay': 0.97,
    'val_part': 0.3,
    'n_in': 3,
    'n_out': 3,
    'dim_in': 64,
    'dim_out': 64,
    'p_coef_y': 1,
    'p_coef_mu': 0.05,
    'p_coef_lambda': 0.0,
    'p_coef_mi': 0.01,
    'loss': 'log',
    'ycf_result': 1,
    'batch_norm': 1,
    'seed': 1,
    'optimizer': 'Adam',
    'output_delay': 100,
    'pred_output_delay': 100,
}

def main():
    print("=" * 80)
    print("Running DeR-CFR Twins Dataset Experiment")
    print("=" * 80)
    print(f"Python Interpreter: {sys.executable}")
    print(f"Dataset: {TWINS_CONFIG['dataform']}")
    print(f"Experiments: {TWINS_CONFIG['experiments']}")
    print(f"Iterations: {TWINS_CONFIG['iterations']}")
    print("=" * 80)

    cmd = [sys.executable, 'main.py']
    for key, value in TWINS_CONFIG.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}={value}')

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n[SUCCESS] Twins experiment completed!")
        print(f"Results location: {TWINS_CONFIG['outdir']}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed! Return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
