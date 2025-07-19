import LocalDataLoader as ldl
import pandas as pd
import FilesReader as fr
import numpy as np
import utils

start_date, end_date = '20100101', '20241231'
# codes_data = ldl.get_descriptions_of_stock_funds()
# codes_data = codes_data[codes_data['open_close_type'] == '开放式']
codes_data = ldl.get_descriptions_of_mostly_stock_funds()
used_style = ('灵活配置型', '积极配置', '沪港深积极配置', '行业混合', '行业股票',
              '大盘成长股票', '沪港深股票型', '灵活配置型(封闭)', '积极配置型',
              '大盘平衡股票', '沪港深积极配置型(封闭)', '中盘成长股票', '大盘价值股票',
              '普通股票型')
codes_data = codes_data[codes_data['invest_style'].isin(used_style)]

#获取基金净值
# nav_data = ldl.get_nav_adj(start_date, end_date, codes_data['code'].unique())
nav_data = fr.read_nav_data()
fund_returns_daily = nav_data.sort_index().groupby('code').pct_change()
fund_returns_daily.columns = ['fund_returns']
fund_returns_quarterly = utils.get_fund_returns(nav_data, freq='QE')
#获取基金公布的持仓
# stocks_details, stocks_key = ldl.get_fund_stocks(codes_data['code'].unique(), start_date, end_date)
# fund_holds = utils.get_fund_holds(stocks_key, freq='QE')
fund_holds = fr.read_fund_holds()

#获取基金alpha
fama_french_returns = fr.read_fama_french_returns_data()
fund_alphas = utils.get_alpha(fund_returns_daily, fama_french_returns)
fund_alphas_quarterly = fund_alphas.reset_index('code').groupby('code').resample('QE')['alpha'].sum()

#获取Fama-French三因子beta值
fund_betas = utils.get_betas(fund_returns_daily, fama_french_returns)
fund_betas = fund_betas.reset_index('code').groupby('code').resample('QE').last()
fund_betas = fund_betas.drop('code', axis = 1)

#获取基金净资产
fund_nvs = ldl.get_fund_net_assets_from_api(start_date, end_date, codes_data['code'].unique())
funds_nv = fund_nvs.sort_index()
funds_nv = funds_nv.reset_index('code').groupby('code').resample('QE')['nv'].last()
size = funds_nv.rename('size').sort_index() / 1e8
#获取基金净资产季度变动
fund_flows = utils.get_fund_flows(fund_nvs, fund_returns_quarterly, freq='QE')

#获取基金换手率
stock_returns = fr.read_stock_returns()
fund_turnover = utils.get_fund_turnover(fund_holds, stock_returns)

#获取费率数据
fund_expense = ldl.get_fund_expense('20070101', end_date, codes = codes_data['code'].unique())
fund_expense = utils.get_daily_expense(fund_expense, fund_returns_daily.index)
fund_expense = fund_expense.reset_index().set_index('date').groupby('code')['expense'].resample('QE').last()

#获取股票日度收益率数据
stock_returns_daily = fr.read_stock_returns_daily()
stock_MAX_monthly = utils.max_returns(stock_returns_daily)
stock_MAX_quarterly = stock_MAX_monthly.reset_index('stock_code').groupby('stock_code').resample('QE').last()
stock_MAX5_monthly = utils.max_returns_5(stock_returns_daily)
stock_MAX5_quarterly = stock_MAX5_monthly.reset_index('stock_code').groupby('stock_code').resample('QE').last()
#获取基金Lottery权重
fund_MAX_quarterly = utils.get_fund_hold_measures_by_code(fund_holds, stock_MAX_quarterly['MAX'])
fund_MAX5_quarterly = utils.get_fund_hold_measures_by_code(fund_holds, stock_MAX5_quarterly['MAX5'])

#基金收益率Lottery
fund_returns_MAX_quarterly = fund_returns_daily.reset_index('code').groupby('code').resample('QE')['fund_returns'].max()
fund_returns_MAX_quarterly.name = 'MAX_FUND'

#获取returns gap
KSG_returns_gap = utils.KSG_returns_gap(fund_returns_quarterly['fund_returns'], fund_holds, stock_returns)

#所有指标放一起
combined_data = pd.concat([fund_MAX_quarterly, fund_MAX5_quarterly, fund_returns_MAX_quarterly, fund_alphas_quarterly, size, fund_turnover,
                           fund_flows, fund_returns_quarterly, fund_expense, fund_betas, KSG_returns_gap], axis = 1)
combined_data = utils.get_funds_age(codes_data, combined_data)
combined_data = utils.drop_small_and_young(combined_data)
combined_data['fund_returns_next_quarter'] = fund_returns_quarterly.sort_index().groupby('code').shift(-1)
combined_data['alphas_next_quarter'] = combined_data['alpha'].sort_index().groupby('code').shift(-1)
combined_data = combined_data.sort_index()
combined_data = combined_data.dropna(subset = ['MAX'])
result = utils.regress_alphas_on_lottery(combined_data)
result.to_excel('Why Do Mutual Funds Hold Lottery Stocks/下一期alpha对彩票股因子回归.xlsx')
combined_data['flow_next_quarter'] = combined_data.groupby('code')['flow'].shift(-1)
result = utils.regress_flows_on_nonlinear_alphas(combined_data)
result.to_excel('Why Do Mutual Funds Hold Lottery Stocks/下一期资金流入对彩票股因子回归.xlsx')
result = utils.regress_lottery_factors_on_past(combined_data)
result.to_excel('Why Do Mutual Funds Hold Lottery Stocks/彩票股因子持续性回归.xlsx')
combined_data['lottery_rank'] = combined_data.groupby('date')['MAX'].rank(pct = True) * 100 // 10 + 1
combined_data['lottery_rank'] = combined_data['lottery_rank'].mask(combined_data['lottery_rank'] == 11, 10)
corr = combined_data[['MAX', 'MAX5', 'alpha', 'size', 'age', 'expense', 'turnover', 'flow']].corr()
corr.to_excel('Why Do Mutual Funds Hold Lottery Stocks/相关系数.xlsx')
group_desc_list = []
for c in ['alpha', 'size', 'age', 'expense', 'turnover', 'flow', 'market', 'smb', 'hml']:
    desc = combined_data.groupby(['lottery_rank', 'date'])[c].mean().unstack('lottery_rank').sort_index().dropna().mean().rename(c)
    group_desc_list.append(desc)
group_desc = pd.concat(group_desc_list, axis = 1)
group_desc.to_excel('Why Do Mutual Funds Hold Lottery Stocks/分组描述性统计.xlsx')

returns = combined_data.groupby(['lottery_rank', 'date'])['alphas_next_quarter'].mean().unstack('lottery_rank').sort_index().dropna()
returns.cumsum().to_excel('Why Do Mutual Funds Hold Lottery Stocks/MAX分组alpha累计.xlsx')
returns.mean().to_excel('Why Do Mutual Funds Hold Lottery Stocks/不同分组公募基金未来alpha值.xlsx')

combined_data['MAX_change'] = combined_data.sort_index().groupby('code')['MAX'].diff().groupby('code').shift(-1)
combined_data['year'] = combined_data.index.get_level_values('date').year
combined_data['cum_returns'] = combined_data.sort_index().groupby(['code', 'year'])['fund_returns'].transform('cumsum')
combined_data['year_returns_isna'] = combined_data['fund_returns'].isna()
combined_data['cum_returns'] = combined_data['cum_returns'].mask(combined_data.sort_index().groupby(['code', 'year'])['year_returns_isna'].transform('any'), np.nan)
combined_data['quarter'] = combined_data.index.get_level_values('date').month // 3
combined_data['cum_returns_rank'] = combined_data['cum_returns'].groupby('date').rank(pct = True) * 100 // 20 + 1
combined_data['cum_returns_rank'] = combined_data['cum_returns_rank'].mask(combined_data['cum_returns_rank'] == 6, 5)
desc = combined_data.groupby(['quarter', 'cum_returns_rank'])['MAX_change'].mean().unstack('cum_returns_rank')
desc.to_excel('Why Do Mutual Funds Hold Lottery Stocks/不同季度不同历史业绩彩票股因子值变动.xlsx')