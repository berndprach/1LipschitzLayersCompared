import pandas
import matplotlib.pyplot as plt
from itertools import filterfalse
from math import ceil
from warnings import warn


class Statistics:
    vars_aggs: dict
    vars_noaggs: dict
    data: pandas.DataFrame

    def __init__(self, vars_aggs: dict[str: str]) -> None:
        r'''
            vars_aggs: contains a dictionary of the desired variables and how
            they have to be aggregated.
            If the aggregation is 'none' than no aggregation is performed, make sure
            only 1 dimensional array has this property

            Example
            `Statistics(vars_aggs={'epoch': 'none',
                                   'train_accuracy':'mean'
                                   'train_margin':'median'
                                   })` 

            Methods:
            - update: updates the statistics with the given values computing the
            aggregation provided in vars_aggs
            - get_last: returns the last computed statistics
            - save: saves the statistics in a csv file
            - save_plot: saves the statistics in a plot, if subplots is True
            then same metrics belonging to different split (train or val) are 
            plotted togheter

        '''

        def not_requires_agg(args): return args[1] == 'none'
        def requires_lambda_agg(args): return 'lambda' in args[1]
        def requires_str_agg(args): return not not_requires_agg(
            args) and not requires_lambda_agg(args)

        vars_aggs_str = dict(filter(requires_str_agg, vars_aggs.items()))
        vars_aggs_lambda = dict(filter(requires_lambda_agg, vars_aggs.items()))
        vars_aggs_lambda = {k: eval(v) for k, v in vars_aggs_lambda.items()}
        self.vars_aggs = {**vars_aggs_str, **vars_aggs_lambda}
        self.vars_noaggs = dict(filter(not_requires_agg, vars_aggs.items()))
        self.data = pandas.DataFrame(columns=vars_aggs.keys())

    def requires_agg(self, args):
        k, v = args
        return k in self.vars_aggs.keys()

    def to_series(self, args):
        k, v = args
        return k, pandas.Series(v)

    def update(self, **kwargs) -> None:
        # values in kwargs that should be aggregated
        values_agg = filter(self.requires_agg, kwargs.items())
        values_agg = dict(map(self.to_series, values_agg))
        # values in kwargs that should not be aggregated
        values_noagg = dict(filterfalse(self.requires_agg, kwargs.items()))

        # aggregating kwargs that requires aggregation
        def isin_kwargs(args): return args[0] in kwargs.keys()
        agg_maps = dict(filter(isin_kwargs, self.vars_aggs.items()))
        # Union of the set of aggregated kwargs

        values_dict = pandas.DataFrame(values_agg).agg(agg_maps).to_dict()
        values_dict = {**values_noagg, **values_dict}
        # Appending to the exisiting data with timestamp index
        idx = [pandas.Timestamp.now()]
        df_actual = pandas.DataFrame(values_dict, index=idx)
        self.data = pandas.concat([self.data, df_actual],)

    def get_last(self):
        return self.data.iloc[-1, :].to_dict()

    def save(self, path: str):
        self.data.to_csv(path)

    def save_plot(self, path: str, subplots=True):
        if pandas.__version__ != '1.5.3':
            return None

        cols = self.data.columns
        train_metrics = [name for name in cols if 'train' in name]
        val_metrics = [name for name in cols if 'val' in name]
        train_metrics.sort()
        val_metrics.sort()
        plot_y = list(zip(train_metrics, val_metrics))
        if not subplots:
            warn('Warning: subplots=False has not been implemented yet.')
        # fit the subplots into a square layout
        n_plots = 1+len(plot_y)
        n_rows = ceil(n_plots**0.5)
        n_cols = ceil(n_plots / n_rows)

        ax = self.data.plot(subplots=plot_y, layout=(n_rows, n_cols))

        plt.savefig(path)

    def __repr__(self) -> str:
        vars_aggs = {**self.vars_aggs, **self.vars_noaggs}
        repr = 'Metric Aggreggation Map:\n'
        repr += str(vars_aggs).replace(',', '\n') + '\n'
        return repr + self.data.__repr__()
