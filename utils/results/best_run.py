import pandas as pd


def get_best_lr_wd(df_stats: pd.DataFrame) -> pd.DataFrame:
    r'''
    Given a pandas.DataFrame with the statistics of the runs, create a yml file
    with the best lr and wd for each model.
    '''
    df = df_stats.copy().dropna(axis=0)
    df.reset_index(inplace=True, drop=True)
    # Change LOT2t with LOT
    df.get_conv = df.get_conv.str.replace('2t', '')

    df_max = df.loc[:, ['get_conv', 'model_id', 'val_robstacc_36']].groupby(['model_id', 'get_conv']
                                                                            ).idxmax().reset_index().val_robstacc_36
    df_best = df.iloc[df_max.values, :].loc[:, ['model_id',
                                                'get_conv',
                                                'lr',
                                                'weight_decay',
                                                'val_robstacc_36']]
    # Rename AOLConv2dDirac to AOLConv2d and same for BnB
    df_best['get_conv'] = df_best.get_conv.str.replace(
        'AOLConv2dDirac', 'AOLConv2d')
    df_best['get_conv'] = df_best.get_conv.str.replace(
        'BnBConv2dDirac', 'BnBConv2d')
    # Creating a pivot table
    return df_best
