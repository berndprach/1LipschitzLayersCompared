import os

import pandas

from utils.parsers import SettingParser


def get_hyperparams(run_dir: str, hyper_params: list) -> dict:
    fname = run_dir+'/dumped_setting.yml'
    settings = SettingParser(fname)
    params_values = map(settings.__item__, hyper_params)
    return dict(zip(hyper_params, params_values))


def get_metrics(run_dir: str, metrics: list) -> dict:
    fname = run_dir+'/training_statistics.csv'
    if os.path.exists(fname):
        df = pandas.read_csv(fname)
        metrics_dict = df.loc[:, metrics].iloc[[-1], :].to_dict(orient='list')
        return metrics_dict
    else:
        print(f'Not enough statistics for run {run_dir}')
        return dict(zip(metrics, [[None]]*len(metrics)))


def get_statistics(root_dir: str, metrics: list, hyper_params: list,
                   only_completed=True, mode='random_search') -> pandas.DataFrame:
    r'''
    Read the statistics of the runs in root_dir and return a pandas.DataFrame with
    the desired metrics and hyperparameters.
    '''
    assert mode in ['random_search', 'final_training']
    run_directories = [dir_path for dir_path, _, _ in os.walk(root_dir)]
    df = pandas.DataFrame(columns=metrics+hyper_params)
    for run_dir in run_directories:
        try:
            stats_dict = get_hyperparams(run_dir, hyper_params)
            stats_dict.update(**get_metrics(run_dir, metrics))
            run_dname = run_dir.split('/')[-1]
            stats_dict.update(run_dname=run_dname)
            df = pandas.concat(
                [df, pandas.DataFrame(stats_dict)], ignore_index=True)
        except FileNotFoundError:
            pass
    # Creating addictional columns
    df['start_time'] = pandas.to_datetime(df.run_dname.apply(
        lambda x: x.split('.')[0]), format='%Y%m%d_%H%M%S')

    # Keep only completed runs
    df['completed'] = (df.epoch == df.epochs)
    if only_completed:
        df = df[df.completed]

    if mode == 'final_training':
        # for each couple model, get_conv, keep only the most recent
        df = df.sort_values(by=['model_id', 'get_conv', 'start_time'])
        df = df.drop_duplicates(subset=['model_id', 'get_conv'], keep='last')
    df.reset_index(drop=True, inplace=True)
    return df
