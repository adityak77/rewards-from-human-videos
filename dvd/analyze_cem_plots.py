import argparse
from collections import defaultdict
import os
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cem_dir', type=str, default='./cem_plots/', help='Directory containing CEM runs')

    args = parser.parse_args()

    results_last = defaultdict(list)
    results_any_timestep = defaultdict(list)
    for file in os.listdir(args.cem_dir):
        if file.startswith('old_plots'):
            continue
        
        file_prefix = '_'.join(file.split('_')[:-1])

        res_file = os.path.join(args.cem_dir, file, 'results.pkl')
        with open(res_file, 'rb') as f:
            res_dict = pickle.load(f)

        total_iterations = res_dict['total_iterations']
        total_last_success = res_dict['total_last_success']
        total_any_timestep_success = res_dict['total_any_timestep_success']

        if total_iterations < 10:
            print('Not enough iterations')
            continue

        last_success_rate = total_last_success / total_iterations
        any_timestep_success_rate = total_any_timestep_success / total_iterations

        results_last[file_prefix].append(last_success_rate)
        results_any_timestep[file_prefix].append(any_timestep_success_rate)

    print('Last timestep success rate')
    for key in sorted(results_last.keys()):
        val = results_last[key]
        print(key, sum(val) / len(val))
        print(val)

    print('Any timestep success rate')
    for key in sorted(results_any_timestep.keys()):
        val = results_any_timestep[key]
        print(key, sum(val) / len(val))
        print(val)

if __name__ == '__main__':
    main()
