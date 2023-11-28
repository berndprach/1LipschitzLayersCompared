from argparse import ArgumentParser
from utils.results import get_statistics, get_best_lr_wd
import os

parser = ArgumentParser(
    'Script to generate tables and figures of the paper')

parser.add_argument('--runs-path',
                    type=str,
                    default='.',
                    help='Path to the root directory where runs are stored.')

parser.add_argument('--output-path',
                    default='./best_lr_wd.csv',
                    type=str,
                    help='Path to the output file.')

parser.add_argument('--all-runs',
                    action='store_true',
                    help='If set, even not completed runs are considered.')

# Hyper_parameters to be compared
HYPER_PARAMETERS = ['model_id', 'get_conv', 'lr', 'weight_decay', 'epochs']
# Metrics to be analysed
METRICS = ['val_accuracy', 'val_robstacc_36', 'epoch']


def main():
    args = parser.parse_args()
    runs_path = args.runs_path
    output_path = args.output_path
    only_completed = not args.all_runs
    # Get statistics
    df_stats = get_statistics(runs_path, metrics=METRICS,
                              hyper_params=HYPER_PARAMETERS,
                              mode='random_search',
                              only_completed=only_completed)

    # Get best lr and wd
    df_best = get_best_lr_wd(df_stats)
    print(df_best)
    df_best.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
