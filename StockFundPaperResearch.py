import LocalDataLoader as ldl
import RegressUtils as utils
import pandas as pd
import FilesReader as fr
import numpy as np
import DisplayUtils as du

start_date, end_date = '20100101', '20241231'
# codes_data = ldl.get_descriptions_of_stock_funds()
# codes_data = codes_data[codes_data['open_close_type'] == '开放式']
codes_data = fr.read_stock_fund_codes_data()
#获取基金净值
nav_data = ldl.get_nav_adj(start_date, end_date, codes_data['code'].unique())
fund_returns_daily = nav_data.sort_index().groupby('code').pct_change()
fund_returns_daily.columns = ['fund_returns']
fund_returns_quarterly = utils.get_fund_returns(nav_data, freq='QE')
#获取基金公布的持仓
stocks_details, stocks_key = ldl.get_fund_stocks(codes_data['code'].unique(), start_date, end_date)
fund_holds = utils.get_fund_holds(stocks_key, freq='QE')

#获取基金alpha
fama_french_returns = fr.read_fama_french_returns_data()
fund_alphas = utils.get_alpha(fund_returns_daily, fama_french_returns)
fund_alphas_2_month = utils.get_alpha_2_month(fund_alphas)
fund_alphas_quarterly = fund_alphas.reset_index('code').groupby('code').resample('QE')['alpha'].sum()

#获取基金净资产
fund_nvs = ldl.get_fund_net_assets_from_api(start_date, end_date, codes_data['code'].unique())
funds_nv = fund_nvs.sort_index()
funds_nv = funds_nv.reset_index('code').groupby('code').resample('QE')['nv'].last()
size = np.log(funds_nv).rename('size').sort_index()
#获取基金净资产季度变动
fund_flows = utils.get_fund_flows(fund_nvs, fund_returns_quarterly, freq='QE')
#获取基金换手率数据
# stock_returns = ldl.get_stock_returns(start_date, end_date, freq='QE')
stock_returns = fr.read_stock_returns()
fund_turnover = utils.get_fund_turnover(fund_holds, stock_returns)
#获取窗口粉饰因子数据
BHRG = utils.BHRG(fund_returns_quarterly['fund_returns'], fund_holds, stock_returns)
winners_losers = utils.get_winners_losers(stock_returns['stock_returns'])
winners_losers_proportion = utils.get_winner_loser_proportion(fund_holds, winners_losers)
rank_gap = utils.get_rank_gap(fund_returns_quarterly, winners_losers_proportion)
#获取Manger Skills
KSG_returns_gap = utils.KSG_returns_gap(fund_returns_quarterly['fund_returns'], fund_holds, stock_returns)

#获取费率数据
fund_expense = ldl.get_fund_expense('20070101', end_date, codes = codes_data['code'].unique())
fund_expense = utils.get_daily_expense(fund_expense, fund_returns_daily.index)
fund_expense = fund_expense.reset_index().set_index('date').groupby('code')['expense'].resample('QE').last()

#合并所有数据
combined_data = pd.concat([BHRG, rank_gap, fund_alphas_quarterly, KSG_returns_gap, size, fund_turnover,
                           fund_flows, fund_returns_quarterly, fund_expense, winners_losers_proportion, fund_alphas_2_month], axis = 1)
combined_data['fund_returns_next_quarter'] = fund_returns_quarterly.sort_index().groupby('code').shift(-1)
combined_data['alphas_next_quarter'] = combined_data['alpha'].sort_index().groupby('code').shift(-1)
combined_data = combined_data.sort_index()

#结果1 投资者会对公布的持仓作出反应
model = utils.flow_window_dress_regress(combined_data, y_name = 'BHRG')
result_format = du.display_regress_result(model)
result_format.to_excel('Window Dress in Mutual Funds/基金资金净流入与持仓公布之间的关系.xlsx')

#进行窗口装饰的共同基金一般具有哪些特征？
combined_data['Manager Skill'] = combined_data.groupby('date')['KSG_returns_gap'].rank(pct = True) * 100 // 20 + 1
combined_data['Manager Skill'] = combined_data['Manager Skill'].mask(combined_data['Manager Skill'] == 6, 5)
combined_data['2-Month Alpha group'] = combined_data.groupby(['date', 'Manager Skill'])['2-Month Alpha'].rank(pct = True) * 100 // 20 + 1
combined_data['2-Month Alpha group'] = combined_data['2-Month Alpha group'].mask(combined_data['2-Month Alpha group'] == 6, 5)
group_desc_BHRG = combined_data.groupby(['Manager Skill', '2-Month Alpha group'])['BHRG'].mean().unstack('2-Month Alpha group')

combined_data['Manager Skill'] = combined_data.groupby('date')['KSG_returns_gap'].rank(pct = True) * 100 // 20 + 1
combined_data['Manager Skill'] = combined_data['Manager Skill'].mask(combined_data['Manager Skill'] == 6, 5)
combined_data['2-Month Alpha group'] = combined_data.groupby(['date', 'Manager Skill'])['2-Month Alpha'].rank(pct = True) * 100 // 20 + 1
combined_data['2-Month Alpha group'] = combined_data['2-Month Alpha group'].mask(combined_data['2-Month Alpha group'] == 6, 5)
group_desc_rank_gap = combined_data.groupby(['Manager Skill', '2-Month Alpha group'])['rank_gap'].mean().unstack('2-Month Alpha group')
writer = pd.ExcelWriter('Window Dress in Mutual Funds/窗口装饰按管理能力与历史alpha分组统计.xlsx')
group_desc_BHRG.to_excel(writer, sheet_name='BHRG')
group_desc_rank_gap.to_excel(writer, sheet_name = 'rank')
writer.close()
model = utils.window_dress_fund_characters_regress(combined_data, WD = 'BHRG')
result_format_BHRG = du.display_regress_result(model)
model = utils.window_dress_fund_characters_regress(combined_data, WD = 'rank_gap')
result_format_rank_gap = du.display_regress_result(model)
writer = pd.ExcelWriter('Window Dress in Mutual Funds/窗口装饰对基金特征指标回归结果.xlsx')
result_format_BHRG.to_excel(writer, sheet_name='BHRG')
result_format_rank_gap.to_excel(writer, sheet_name = 'rank')
writer.close()

#WD与动量因子的相关性
stock_returns_monthly = fr.read_stock_returns_monthly()
stock_returns_daily = fr.read_stock_returns_daily()
# fund_holds_winners_returns_daily = utils.get_daily_winners_returns(stock_returns_daily['stock_returns'], stock_returns_monthly['stock_returns'], fund_holds)
fund_holds_winners_returns_daily = fr.read_holds_winners_returns()
funds_monthly_holds_winners_corr = utils.get_funds_monthly_holds_winners_corr(fund_returns_daily, fund_holds_winners_returns_daily)
model_BHRG = utils.corr_window_dressing_regress(funds_monthly_holds_winners_corr, BHRG.rename('WD'))
result_format_BHRG = du.display_regress_result(model_BHRG)
model_rank_gap = utils.corr_window_dressing_regress(funds_monthly_holds_winners_corr, rank_gap.rename('WD'))
result_format_rank_gap = du.display_regress_result(model_rank_gap)
result_format = pd.concat([result_format_BHRG, result_format_rank_gap], axis = 1)
result_format.columns = ['BHRG', 'Rank Gap']
result_format.to_excel('Window Dress in Mutual Funds/相关系数对窗饰因子回归结果.xlsx')

#窗口装饰与基金未来业绩之间的关系
combined_data['BHRG_rank'] = combined_data['BHRG'].groupby('date').rank(pct = True) * 100 // 10 + 1
combined_data['BHRG_rank'] = combined_data['BHRG_rank'].mask(combined_data['BHRG_rank'] == 11, 10)
future_alphas_BHRG = combined_data.groupby('BHRG_rank')['alphas_next_quarter'].mean() * 4
combined_data['Rank_Gap_rank'] = combined_data['rank_gap'].groupby('date').rank(pct = True) * 100 // 10 + 1
combined_data['Rank_Gap_rank'] = combined_data['Rank_Gap_rank'].mask(combined_data['Rank_Gap_rank'] == 11, 10)
future_alphas_Rank_Gap = combined_data.groupby('Rank_Gap_rank')['alphas_next_quarter'].mean() * 4
future_alphas_desc = pd.concat([future_alphas_BHRG, future_alphas_Rank_Gap], axis = 1)
future_alphas_desc.columns = ['BHRG', 'Rank Gap']
future_alphas_desc.to_excel('Window Dress in Mutual Funds/未来alpha对窗饰因子分组结果.xlsx')
model_BHRG = utils.future_alpha_window_dressing_regress(combined_data, 'BHRG')
result_format_BHRG = du.display_regress_result(model_BHRG)
model_rank_gap = utils.future_alpha_window_dressing_regress(combined_data, 'rank_gap')
result_format_rank_gap = du.display_regress_result(model_rank_gap)
result_format = pd.concat([result_format_BHRG, result_format_rank_gap], axis = 1)
result_format.columns = ['BHRG', 'Rank Gap']
result_format.to_excel('Window Dress in Mutual Funds/未来业绩对窗饰因子回归结果.xlsx')


corr = combined_data.groupby('date')[['BHRG', 'fund_returns_next_quarter']].corr().iloc[::2, 1].droplevel(1)
corr2 = combined_data.groupby('date')[['BHRG', 'fund_returns']].corr().iloc[::2, 1].droplevel(1)
combined_data['BHRG_rank'] = combined_data['BHRG'].groupby('date').rank(pct = True) * 100 // 10 + 1
combined_data['BHRG_rank'] = combined_data['BHRG_rank'].mask(combined_data['BHRG_rank'] == 11, 10)
combined_data.groupby(['date', 'BHRG_rank'])['fund_returns_next_quarter'].mean().groupby('BHRG_rank').mean() * 4

corr_BHRG_alpha = combined_data.groupby('date')[['BHRG', 'alphas_next_quarter']].corr().iloc[::2, 1].droplevel(1)
corr2 = combined_data.groupby('date')[['BHRG', 'alpha']].corr().iloc[::2, 1].droplevel(1)
combined_data['BHRG_rank'] = combined_data['BHRG'].groupby('date').rank(pct = True) * 100 // 10 + 1
combined_data['BHRG_rank'] = combined_data['BHRG_rank'].mask(combined_data['BHRG_rank'] == 11, 10)
combined_data.groupby('BHRG_rank')['alphas_next_quarter'].mean() * 4

corr_KSG_alpha = combined_data.groupby('date')[['KSG_returns_gap', 'alphas_next_quarter']].corr().iloc[::2, 1].droplevel(1)
corr2 = combined_data.groupby('date')[['KSG_returns_gap', 'alpha']].corr().iloc[::2, 1].droplevel(1)

combined_data['Manager Skill'] = combined_data.groupby('date')['KSG_returns_gap'].rank(pct = True) * 100 // 20 + 1
combined_data['Manager Skill'] = combined_data['Manager Skill'].mask(combined_data['Manager Skill'] == 6, 5)
combined_data['2-Month Alpha'] = combined_data.groupby(['date', 'Manager Skill'])['2-Month Alpha'].rank(pct = True) * 100 // 20 + 1
combined_data['2-Month Alpha'] = combined_data['2-Month Alpha'].mask(combined_data['2-Month Alpha'] == 6, 5)
combined_data.groupby(['Manager Skill', '2-Month Alpha'])['BHRG'].mean().unstack('2-Month Alpha')

combined_data = pd.concat([BHRG, KSG_returns_gap, fund_alphas_quarterly], axis = 1)
combined_data['Manager Skill'] = combined_data.groupby('date')['KSG_returns_gap'].rank(pct = True) * 100 // 20 + 1
combined_data['Manager Skill'] = combined_data['Manager Skill'].mask(combined_data['Manager Skill'] == 6, 5)
combined_data['3-Month Alpha'] = combined_data.groupby(['date', 'Manager Skill'])['alpha'].rank(pct = True) * 100 // 20 + 1
combined_data['3-Month Alpha'] = combined_data['3-Month Alpha'].mask(combined_data['3-Month Alpha'] == 6, 5)
combined_data.groupby(['Manager Skill', '3-Month Alpha'])['BHRG'].mean().unstack('3-Month Alpha')