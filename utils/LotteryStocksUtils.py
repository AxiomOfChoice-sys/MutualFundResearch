import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS
def max_returns(stock_returns_daily):
    stock_returns_daily = stock_returns_daily.reset_index('stock_code')
    MAX = stock_returns_daily.groupby('stock_code').resample('ME')['stock_returns'].max().rename('MAX')
    return MAX

def max_returns_5(stock_returns_daily):
    stock_returns_daily = stock_returns_daily.reset_index('stock_code')
    MAX5 = stock_returns_daily.groupby('stock_code').resample('ME')['stock_returns'].apply(lambda s:s.sort_values(ascending=False).iloc[:5].mean()).rename('MAX5')
    return MAX5

def get_fund_hold_measures(fund_holds, stock_measures):
    name = stock_measures.name
    fund_holds = fund_holds / fund_holds.groupby(['stock_code', 'date']).sum()
    fund_holds = fund_holds.reset_index('stock_code').swaplevel().sort_index()
    combined_data = fund_holds.join(stock_measures).set_index('stock_code', append=True)
    fund_hold_measures = (combined_data['weights'] * combined_data[name]).groupby(['stock_code', 'date']).sum()
    fund_hold_measures.name = name
    return fund_hold_measures

def get_fund_hold_measures_by_code(fund_holds, stock_measures):
    name = stock_measures.name
    fund_holds = fund_holds / fund_holds.groupby(['code', 'date']).sum()
    data_list = []
    for code, group in fund_holds.groupby('code'):
        group = group.reset_index('code').swaplevel().sort_index()
        combined_data = group.join(stock_measures).set_index('code', append=True)
        fund_hold_measures = (combined_data['weights'] * combined_data[name]).groupby(['code', 'date']).sum()
        fund_hold_measures.name = name
        data_list.append(fund_hold_measures)
        print(code)
    return pd.concat(data_list)

def cross_section_z_score(s):
    z_score = (s - s.groupby('date').mean()) / s.groupby('date').std()
    return z_score

def regress_alphas_on_lottery(combined_data):
    combined_data2 = combined_data[['alphas_next_quarter', 'MAX', 'MAX_FUND', 'alpha', 'size', 'age',
                                    'expense', 'turnover', 'flow', 'smb', 'hml', 'KSG_returns_gap']]
    combined_data2['size'] = np.log(combined_data2['size'])
    combined_data2['age'] = np.log(combined_data2['age']).replace([np.inf, -np.inf], [np.nan, np.nan])
    combined_data2 = sm.add_constant(combined_data2)
    def get_regress_result(df):
        y = df['alphas_next_quarter']
        X = df[['const', 'MAX', 'MAX_FUND', 'alpha', 'size', 'age', 'expense', 'turnover', 'flow', 'smb', 'hml', 'KSG_returns_gap']]
        model = sm.OLS(y, X, missing='drop').fit()
        return model.params
    params = combined_data2.dropna().groupby('date').apply(get_regress_result)
    result = pd.concat([params.mean(), params.mean() / params.std()], axis = 1)
    result.columns = ['params', 't-values']
    return result

def regress_flows_on_nonlinear_alphas(combined_data):
    combined_data2 = combined_data[['flow_next_quarter', 'alpha', 'MAX', 'size', 'age', 'expense', 'turnover']]
    y = combined_data2['flow_next_quarter']
    X = combined_data2[['MAX', 'alpha', 'size', 'age', 'expense', 'turnover']]
    X['alpha_rank'] = X['alpha'].groupby('date').rank(pct = True)
    X['low'] = X['alpha_rank'].mask(X['alpha_rank'] > 0.2, 0.2)
    X['mid'] = (X['alpha_rank'] - X['low']).mask(X['alpha_rank'] - X['low'] > 0.6, 0.6)
    X['high'] = X['alpha_rank'] - X['low'] - X['mid']
    X = X[['MAX', 'low', 'mid', 'high', 'size', 'age', 'expense', 'turnover']]
    X['size'] = np.log(X['size'])
    X['age'] = np.log(X['age']).replace([np.inf, -np.inf], [np.nan, np.nan])
    X = sm.add_constant(X)
    for c in ['MAX', 'size', 'age', 'expense', 'turnover']:
        X[c] = cross_section_z_score(X[c])
    model = PanelOLS(y, X, entity_effects=True, time_effects=True).fit()
    result = pd.concat([model.params, model.pvalues], axis = 1)
    result.columns = ['params', 'p']
    return result


def regress_lottery_factors_on_past(combined_data):
    combined_data2 = combined_data[['MAX', 'MAX_FUND', 'alpha', 'size', 'age',
                                    'expense', 'turnover', 'flow', 'smb', 'hml', 'KSG_returns_gap']]
    combined_data2['size'] = np.log(combined_data2['size'])
    combined_data2['age'] = np.log(combined_data2['age']).replace([np.inf, -np.inf], [np.nan, np.nan])
    combined_data2 = sm.add_constant(combined_data2)

    def get_regress_result(df):
        y = df['MAX_FUND_future']
        X = df[['const', 'MAX', 'MAX_FUND', 'alpha', 'size', 'age', 'expense', 'turnover', 'flow', 'smb', 'hml',
                'KSG_returns_gap']]
        model = sm.OLS(y, X, missing='drop').fit()
        return model.params
    result_list = []
    for i in range(1, 5):
        combined_data2['MAX_FUND_future'] = combined_data['MAX_FUND'].sort_index().groupby('code').shift(-i)
        params = combined_data2.dropna().groupby('date').apply(get_regress_result)
        result = pd.concat([params.mean(), params.mean() / params.std()], axis=1)
        result.columns = [f'params-{i}', f't-values-{i}']
        result_list.append(result)
    result = pd.concat(result_list, axis = 1)
    return result
