import statsmodels.api as sm
import pandas as pd
from statsmodels.regression.rolling import RollingOLS

def regress_month_quarter_end_returns(returns_data):
    returns_data['is_month_end'] = returns_data['is_quarter_end'] = returns_data['is_year_end'] = 0
    dates = returns_data.index.get_level_values('date').unique()
    date_series = pd.Series(dates, index=dates).sort_index()
    month_end_dates = date_series.resample('ME').last()
    quarter_end_dates = date_series.resample('QE').last()
    year_end_dates = date_series.resample('YE').last()
    returns_data['is_month_end'] = returns_data['is_month_end'].mask(
        returns_data.index.get_level_values('date').isin(month_end_dates), 1)
    returns_data['is_quarter_end'] = returns_data['is_quarter_end'].mask(
        returns_data.index.get_level_values('date').isin(quarter_end_dates), 1)
    returns_data['is_year_end'] = returns_data['is_year_end'].mask(
        returns_data.index.get_level_values('date').isin(year_end_dates), 1)
    X = returns_data[['is_month_end', 'is_quarter_end', 'is_year_end']]
    X = sm.add_constant(X)
    y = returns_data['returns']
    model = sm.OLS(y, X, missing = 'drop').fit()
    return model

def regress_month_quarter_start_returns(returns_data):
    if len(returns_data) < 50:return
    returns_data['is_month_end'] = returns_data['is_quarter_end'] = returns_data['is_year_end'] = 0
    dates = returns_data.index.get_level_values('date').unique()
    date_series = pd.Series(dates, index=dates).sort_index()
    month_end_dates = date_series.resample('ME').last()
    quarter_end_dates = date_series.resample('QE').last()
    year_end_dates = date_series.resample('YE').last()
    returns_data['is_month_end'] = returns_data['is_month_end'].mask(
        returns_data.index.get_level_values('date').isin(month_end_dates), 1)
    returns_data['is_quarter_end'] = returns_data['is_quarter_end'].mask(
        returns_data.index.get_level_values('date').isin(quarter_end_dates), 1)
    returns_data['is_year_end'] = returns_data['is_year_end'].mask(
        returns_data.index.get_level_values('date').isin(year_end_dates), 1)
    returns_data['is_month_start'] = returns_data['is_quarter_start'] = returns_data['is_year_start'] = 0
    dates = returns_data.index.get_level_values('date').unique()
    date_series = pd.Series(dates, index=dates).sort_index()
    month_start_dates = date_series.resample('ME').first()
    quarter_start_dates = date_series.resample('QE').first()
    year_start_dates = date_series.resample('YE').first()
    returns_data['is_month_start'] = returns_data['is_month_start'].mask(
        returns_data.index.get_level_values('date').isin(month_start_dates), 1)
    returns_data['is_quarter_start'] = returns_data['is_quarter_start'].mask(
        returns_data.index.get_level_values('date').isin(quarter_start_dates), 1)
    returns_data['is_year_start'] = returns_data['is_year_start'].mask(
        returns_data.index.get_level_values('date').isin(year_start_dates), 1)
    X = returns_data[['is_month_start', 'is_quarter_start', 'is_year_start', 'is_month_end', 'is_quarter_end', 'is_year_end']]
    X = sm.add_constant(X)
    y = returns_data['returns']
    model = sm.OLS(y, X, missing = 'drop').fit()
    return model.params

def get_fund_flows(funds_nv, fund_returns, freq = 'QE'):
    funds_nv = funds_nv.sort_index()
    funds_nv = funds_nv.reset_index('code').groupby('code').resample(freq)['nv'].last()
    fund_flows = funds_nv / funds_nv.groupby('code').shift(1) - (fund_returns['fund_returns'] + 1)
    fund_flows.name = 'flow'
    return fund_flows

def get_fund_returns(nav_adj, freq = 'QE'):
    nav_adj = nav_adj.sort_index()
    nav_adj = nav_adj.reset_index().set_index('date').groupby('code').resample(freq).last()[['f_nav_adjusted']]
    fund_returns = nav_adj.sort_index().groupby('code').pct_change()
    fund_returns.columns = ['fund_returns']
    return fund_returns

def get_fund_holds(fund_holds_raw, freq = 'QE'):
    fund_holds = fund_holds_raw.set_index(['code', 'report_date', 'stock_code'])['ratio']
    fund_holds.index.names = ['code', 'date', 'stock_code']
    fund_holds.name = 'weights'
    if freq == 'QE':
        fund_holds = fund_holds[fund_holds.index.get_level_values('date').is_quarter_end]
    elif freq == 'half_year':
        fund_holds = fund_holds[fund_holds.index.get_level_values('date').is_month_end]
        fund_holds = fund_holds[fund_holds.index.get_level_values('date').month.isin((6, 12))]
    return fund_holds
def get_fund_hold_returns(fund_holds, stock_returns, backward = True):
    fund_holds = fund_holds / fund_holds.groupby(['code', 'date']).sum()
    fund_holds = fund_holds.reset_index('code').swaplevel().sort_index()
    combined_data = fund_holds.join(stock_returns).set_index('code', append = True)
    if backward:
        combined_data['weights'] = combined_data['weights'] / (1 + combined_data['stock_returns'])
        combined_data['weights'] = combined_data['weights'] / combined_data['weights'].groupby('date').sum()
    fund_hold_returns = (combined_data['weights'] * combined_data['stock_returns']).groupby(['code', 'date']).sum()
    fund_hold_returns.name = 'fund_hold_returns'
    return fund_hold_returns

def BHRG(fund_returns, fund_holds, stock_returns):
    fund_hold_returns = get_fund_hold_returns(fund_holds, stock_returns, backward=True)
    BHRG = fund_hold_returns - fund_returns
    BHRG.name = 'BHRG'
    return BHRG

def KSG_returns_gap(fund_returns, fund_holds, stock_returns):
    fund_holds = fund_holds.reset_index()
    fund_holds['date'] = fund_holds['date'] + pd.tseries.offsets.QuarterEnd()
    fund_holds = fund_holds.set_index(['code', 'date', 'stock_code'])['weights']
    fund_returns_future = get_fund_hold_returns(fund_holds, stock_returns, backward=False)
    returns_gap = fund_returns - fund_returns_future
    returns_gap.name = 'KSG_returns_gap'
    returns_gap = returns_gap.groupby('code').rolling(4, min_periods=1).mean().droplevel(0)
    return returns_gap

def get_fund_turnover(fund_holds, stock_returns):
    fund_holds2 = fund_holds.reset_index()
    fund_holds2['date'] = fund_holds2['date'] + pd.tseries.offsets.QuarterEnd()
    fund_holds2 = fund_holds2.set_index(['code', 'date', 'stock_code'])
    fund_holds3 = fund_holds.reset_index()
    fund_holds3['date'] = fund_holds3['date'] - pd.tseries.offsets.QuarterEnd()
    fund_holds3 = fund_holds3.set_index(['code', 'date', 'stock_code'])
    used_index = fund_holds.index.union(fund_holds2.index).union(fund_holds3.index).unique()
    fund_holds = fund_holds.reindex(used_index).sort_index().fillna(0)
    fund_holds = fund_holds.reset_index().set_index(['stock_code', 'date']).join(stock_returns, how = 'left')
    fund_holds['weights2'] = fund_holds['weights'] / (1 + fund_holds['stock_returns'])
    fund_holds = fund_holds.reset_index().set_index(['code', 'date', 'stock_code']).sort_index()
    fund_turnover = fund_holds.dropna().groupby(['code', 'stock_code'])['weights2'].diff().abs().groupby(['code', 'date']).sum() / 2
    fund_turnover.name = 'turnover'
    return fund_turnover

def get_winners_losers(stock_returns):
    pct_rank = stock_returns.groupby('date').rank(pct = True)
    winners_losers = pd.Series([''] * len(pct_rank), index = pct_rank.index)
    winners_losers = winners_losers.mask(pct_rank > 0.8, 'winner')
    winners_losers = winners_losers.mask(pct_rank < 0.2, 'loser')
    winners_losers = winners_losers[winners_losers != '']
    winners_losers.name = 'winners_losers'
    return winners_losers

def get_winner_loser_proportion(fund_holds, winners_losers):
    fund_holds = fund_holds.reset_index().set_index(['stock_code', 'date']).join(winners_losers)
    fund_holds = fund_holds.set_index('code', append = True)
    winner_loser_proportion = fund_holds.groupby(['code', 'date', 'winners_losers'])['weights'].sum().unstack('winners_losers')
    winner_loser_proportion = winner_loser_proportion[['winner', 'loser']]
    winner_loser_proportion.columns = ['winner_proportion', 'loser_proportion']
    return winner_loser_proportion

def get_rank_gap(fund_returns, winner_loser_proportion):
    fund_returns_rank = fund_returns.groupby('date')['fund_returns'].rank(pct = True, ascending = False)
    winner_proportion_rank = winner_loser_proportion['winner_proportion'].groupby('date').rank(pct = True, ascending = False)
    loser_proportion_rank = winner_loser_proportion['loser_proportion'].groupby('date').rank(pct=True, ascending=True)
    rank_gap = (fund_returns_rank - (winner_proportion_rank + loser_proportion_rank) / 2) / 200
    rank_gap.name = 'rank_gap'
    return rank_gap

def get_alpha(fund_returns, factors_returns_daily):
    factors_net_values = (1 + factors_returns_daily.sort_index()).cumprod()
    fund_returns = pd.DataFrame(fund_returns)
    fund_returns.columns = ['fund_returns']
    combined_data = fund_returns.join(factors_net_values).sort_index()
    factors_columns = factors_returns_daily.columns
    for c in factors_columns:
        combined_data[c] = combined_data[c].groupby('code').pct_change()
    def get_rolling_beta(df):
        if len(df) < 60:return
        y = df['fund_returns']
        X = df[factors_columns]
        X = sm.add_constant(X)
        model = RollingOLS(y, X, window = 60).fit()
        return model.params
    betas = combined_data.groupby('code').apply(get_rolling_beta).droplevel(0)
    alphas = fund_returns['fund_returns'] - (combined_data[factors_columns] * betas[factors_columns]).sum(axis = 1)
    alphas.name = 'alpha'
    return alphas

def get_betas(fund_returns, factors_returns_daily):
    factors_net_values = (1 + factors_returns_daily.sort_index()).cumprod()
    fund_returns = pd.DataFrame(fund_returns)
    fund_returns.columns = ['fund_returns']
    combined_data = fund_returns.join(factors_net_values).sort_index()
    factors_columns = factors_returns_daily.columns
    for c in factors_columns:
        combined_data[c] = combined_data[c].groupby('code').pct_change()

    def get_rolling_beta(df):
        if len(df) < 60: return
        y = df['fund_returns']
        X = df[factors_columns]
        X = sm.add_constant(X)
        model = RollingOLS(y, X, window=60).fit()
        return model.params

    betas = combined_data.groupby('code').apply(get_rolling_beta).droplevel(0)
    return betas[factors_columns]

def get_TNA_change(fund_nvs, fund_returns):
    return fund_nvs - (fund_nvs * fund_returns).sort_index().groupby('code').shift(1)

def get_alpha_2_month(alphas):
    alphas_monthly = alphas.reset_index('code').groupby('code').resample('ME')['alpha'].sum()
    def get_first_2(series):
        if len(series) < 3:
            return None
        else:
            return series.iloc[:2].sum()
    return alphas_monthly.reset_index('code').groupby('code').resample('QE')['alpha'].apply(get_first_2).rename('2-Month Alpha')

def get_daily_expense(fund_expense, daily_index):
    used_index = fund_expense.index.union(daily_index).unique()
    fund_expense = fund_expense.reindex(used_index).sort_index()
    fund_expense = fund_expense.groupby('code').ffill()
    return fund_expense

def get_daily_winners_returns(stock_returns_daily, stock_returns_monthly, fund_holds):
    stock_returns_rolling_3_month = stock_returns_monthly.groupby('stock_code').rolling(3, min_periods=1).sum().droplevel(0)
    stock_returns_rank = stock_returns_rolling_3_month.groupby('stock_code').rank(pct = True)
    stock_winners = stock_returns_rank[stock_returns_rank > 0.8]
    fund_holds1 = fund_holds.reset_index()
    fund_holds1['date'] = fund_holds1['date'] - pd.tseries.offsets.MonthEnd()
    fund_holds1 = fund_holds1.set_index(['stock_code', 'date'])
    fund_holds2 = fund_holds.reset_index()
    fund_holds2['date'] = fund_holds2['date'] - pd.tseries.offsets.MonthEnd(2)
    fund_holds2 = fund_holds2.set_index(['stock_code', 'date'])
    fund_holds3 = fund_holds.reset_index()
    fund_holds3['date'] = fund_holds3['date'] - pd.tseries.offsets.MonthEnd(3)
    fund_holds3 = fund_holds3.set_index(['stock_code', 'date'])
    fund_holds_monthly = pd.concat([fund_holds3, fund_holds2, fund_holds1]).sort_index()
    fund_holds_winners_monthly = fund_holds_monthly[fund_holds_monthly.index.isin(stock_winners.index)]
    fund_holds_winners_monthly = fund_holds_winners_monthly.reset_index().set_index(['code', 'date', 'stock_code'])
    fund_holds_winners_monthly = fund_holds_winners_monthly['weights'] / fund_holds_winners_monthly['weights'].groupby(['code', 'date']).transform('sum')
    used_index = fund_holds_winners_monthly.reset_index('code').swaplevel().index.union(stock_returns_daily.index).unique()
    data_list = []
    for code, group in fund_holds_winners_monthly.groupby('code'):
        min_date, max_date = fund_holds_winners_monthly.index.get_level_values('date').min(), fund_holds_winners_monthly.index.get_level_values('date').max()
        group = group.droplevel(['code']).swaplevel()
        used_index_group = used_index[used_index.get_level_values('date') >= min_date]
        used_index_group = used_index_group[used_index_group.get_level_values('date') <= max_date]
        group = group.reindex(used_index_group).sort_index().groupby('stock_code').ffill()
        group['code'] = code
        # data_list.append(group)
        # print(code)
        group.to_csv(f'Window Dress in Mutual Funds/临时数据/{code}.csv')
    # fund_holds_winners_daily = pd.concat(data_list)
    for code, group in fund_holds_winners_monthly.groupby('code'):
        if code < '001692.OF':continue
        codes_data = pd.read_csv(f'Window Dress in Mutual Funds/临时数据/{code}.csv')
        codes_data = codes_data.dropna()
        codes_data['date'] = pd.to_datetime(codes_data['date'])
        codes_data = codes_data.set_index(['stock_code', 'date'])['weights']
        codes_data =pd.DataFrame(codes_data.groupby('stock_code').shift(1)).join(stock_returns_daily).dropna()
        for c in ['weights', 'stock_returns']:
            codes_data[c] = pd.to_numeric(codes_data[c])
        code_holds_winners_returns_daily = pd.DataFrame((codes_data['weights'] * codes_data['stock_returns']).groupby('date').sum().rename('holds_winners_returns'))
        code_holds_winners_returns_daily['code'] = code
        data_list.append(code_holds_winners_returns_daily)
        print(code)

    fund_holds_winners_returns_daily = pd.concat(data_list)
    return fund_holds_winners_returns_daily

def get_funds_monthly_holds_winners_corr(funds_returns_daily, fund_holds_winners_returns_daily):
    winners_net_values = (1 + fund_holds_winners_returns_daily.sort_index()).cumprod()
    fund_returns = pd.DataFrame(funds_returns_daily)
    fund_returns.columns = ['fund_returns']
    combined_data = fund_returns.join(winners_net_values).sort_index()
    combined_data['holds_winners_returns2'] = combined_data['holds_winners_returns'].sort_index().groupby('code').pct_change()
    combined_data['holds_winners_returns2'] = combined_data['holds_winners_returns2'].fillna(combined_data['holds_winners_returns'] - 1)
    combined_data = combined_data.reset_index('code')
    funds_monthly_holds_winners_corr = combined_data.groupby(['code', lambda x:x.strftime('%Y%m01')])[['fund_returns', 'holds_winners_returns2']].corr().iloc[::2, 1]
    funds_monthly_holds_winners_corr.name = 'funds_monthly_holds_winners_corr'
    funds_monthly_holds_winners_corr = funds_monthly_holds_winners_corr.reset_index()
    funds_monthly_holds_winners_corr['date'] = pd.to_datetime(funds_monthly_holds_winners_corr['date']) + pd.tseries.offsets.MonthEnd()
    return funds_monthly_holds_winners_corr.set_index(['code', 'date'])['funds_monthly_holds_winners_corr']

def flow_window_dress_regress(combined_data):
    y = combined_data['flow'].groupby('code').shift(-1)
    X = combined_data[['winner_proportion', 'loser_proportion', 'alpha', 'KSG_returns_gap', 'size', 'turnover']]
    X['alpha'] = X['alpha'].groupby('code').shift(1)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def window_dress_fund_characters_regress(combined_data, WD = 'BHRG'):
    y = combined_data[WD]
    X = combined_data[['2-Month Alpha', 'KSG_returns_gap', 'expense', 'size', 'turnover']]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c])
    X['KSG_returns_gap'] = X['KSG_returns_gap'].groupby('code').shift(1)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def corr_window_dressing_regress(funds_monthly_holds_winners_corr, WD_factors):
    WD_factors1 = WD_factors.reset_index('date')
    WD_factors1['date'] = WD_factors1['date'] - pd.tseries.offsets.MonthEnd(1)
    WD_factors1 = WD_factors1.set_index('date', append = True).iloc[:, 0]
    WD_factors2 = WD_factors.reset_index('date')
    WD_factors2['date'] = WD_factors2['date'] - pd.tseries.offsets.MonthEnd(2)
    WD_factors2 = WD_factors2.set_index('date', append = True).iloc[:, 0]
    WD_factors_monthly = pd.concat([WD_factors, WD_factors1, WD_factors2])
    combined_data = pd.concat([funds_monthly_holds_winners_corr, WD_factors_monthly], axis = 1).sort_index()
    y = combined_data['funds_monthly_holds_winners_corr']
    X = combined_data[[WD_factors.name]]
    import numpy as np
    X['FQEM'] = np.where(X.index.get_level_values('date').month // 3 == 0, 1, 0)
    X['FQEM*WD'] = X['FQEM'] * X[WD_factors.name]
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def future_alpha_window_dressing_regress(combined_data, WD = 'BHRG'):
    y = combined_data['alphas_next_quarter']
    X = combined_data[[WD, 'alpha', 'fund_returns', 'KSG_returns_gap', 'expense', 'size', 'turnover', 'flow']]
    import numpy as np
    X[WD] = np.where(X[WD].groupby('date').rank(pct = True) > 0.9, 1, 0)
    X['fund_returns'] = np.where(X['fund_returns'].groupby('date').rank(pct=True) < 0.1, 1, 0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    return model