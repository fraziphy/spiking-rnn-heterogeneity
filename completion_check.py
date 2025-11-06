import glob
import os
import pickle
import numpy as np

def check_nan_in_pkl(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        def contains_nan(x):
            if isinstance(x, np.ndarray):
                return np.isnan(x).any()
            elif isinstance(x, dict):
                return any(contains_nan(v) for v in x.values())
            elif isinstance(x, (list, tuple)):
                return any(contains_nan(v) for v in x)
            return False
        return contains_nan(data)
    except Exception:
        return True  # Consider unreadable files as problematic

def main():
    log_dirs = glob.glob('./sweep/logs_*/job_*/')

    summary = {
        'ok': 0,
        'stderr_not_empty': [],
        'missing_completion': [],
        'nan_values': [],
    }

    for job_dir in log_dirs:
        errpath = os.path.join(job_dir, 'stderr.log')
        outpath = os.path.join(job_dir, 'stdout.log')

        err_empty = os.path.exists(errpath) and os.path.getsize(errpath) == 0

        pkl_paths = []
        out_done = False
        if os.path.exists(outpath):
            with open(outpath) as f:
                for line in f:
                    if 'Results saved:' in line:
                        parts = line.strip().split()
                        if len(parts) > 2:
                            pkl_paths.append(parts[-1])
            out_done = len(pkl_paths) > 0

        nan_found = False
        for pkl_file in pkl_paths:
            if os.path.exists(pkl_file):
                if check_nan_in_pkl(pkl_file):
                    nan_found = True
                    break
            else:
                nan_found = True
                break

        if err_empty and out_done and not nan_found:
            summary['ok'] += 1
        else:
            if not err_empty:
                summary['stderr_not_empty'].append(job_dir)
            if not out_done:
                summary['missing_completion'].append(job_dir)
            if nan_found:
                summary['nan_values'].append(job_dir)

    # Print concise summary
    print("\nSimulation Sweep Summary:")
    print(f"  Successful jobs: {summary['ok']}")
    print(f"  Jobs with non-empty stderr.log: {len(summary['stderr_not_empty'])}")
    print(f"  Jobs missing completion message: {len(summary['missing_completion'])}")
    print(f"  Jobs with NaN values or missing saved files: {len(summary['nan_values'])}")

    # Optionally, print details for failed categories
    if summary['stderr_not_empty']:
        print("\nJobs with errors (stderr.log not empty):")
        for job in summary['stderr_not_empty']:
            print(f"  - {job}")

    if summary['missing_completion']:
        print("\nJobs missing completion output (no 'Results saved:' line):")
        for job in summary['missing_completion']:
            print(f"  - {job}")

    if summary['nan_values']:
        print("\nJobs with NaN values or missing result files:")
        for job in summary['nan_values']:
            print(f"  - {job}")

if __name__ == "__main__":
    main()
