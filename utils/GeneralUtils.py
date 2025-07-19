import pandas as pd
import statsmodels.api as sm
import numpy as np

def regress_factors(portfolio_returns, factors_returns_daily):
    factors_net_values = (1 + factors_returns_daily.sort_index()).cumprod()
    portfolio_returns = pd.DataFrame(portfolio_returns)
    portfolio_returns.columns = ['portfolio_returns']
    combined_data = portfolio_returns.join(factors_net_values).sort_index()
    factors_columns = factors_returns_daily.columns
    for c in factors_columns:
        combined_data[c] = combined_data[c].groupby('code').pct_change()
    def get_regress_summary(df):
        if len(df) < 60:return
        y = df['fund_returns']
        X = df[factors_columns]
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop').fit()
        params = model.params.rename('coef')
        p_values = model.pvalues.rename('p')
        return pd.concat([params, p_values], axis = 1)
    return combined_data.groupby('code').apply(get_regress_summary)


def get_after_start_nav(codes_data, nav_data):
    '''获取基金建仓期后的基金净值'''
    nav_data_list = []
    for code, group in nav_data.groupby('code'):
        info_data = codes_data[codes_data['code'] == code]
        start_date = info_data['start_date'].iloc[0]
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date) + pd.tseries.offsets.Day(93)
            group = group.loc[code][start_date:]
        else:
            group = group.iloc[13:]
        group['code'] = code
        nav_data_list.append(group)
    nav_data = pd.concat(nav_data_list).reset_index().set_index(['code', 'date']).sort_index()
    return nav_data

def get_funds_age(codes_data, infos):
    combined_data = infos.join(codes_data.set_index('code')['start_date'])
    combined_data['age'] = combined_data.index.get_level_values('date') - combined_data['start_date']
    combined_data['age'] = combined_data['age'].apply(lambda x:x.days / 365.0)
    combined_data = combined_data.drop('start_date', axis = 1)
    return combined_data

def drop_small_and_young(data):
    data = data[data['size'] >= 1]
    data = data[data['age'] >= 1]
    return data
