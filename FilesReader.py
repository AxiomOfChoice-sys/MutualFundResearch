import pandas as pd

def read_stock_fund_codes_data():
    codes_data = pd.read_excel('普通股票型基金.xlsx')
    codes_data['code'] = codes_data['证券代码']
    codes_data['name'] = codes_data['证券简称'].str.extract('(.+)[A-Z]')
    codes_data['name'] = codes_data['name'].fillna(codes_data['证券简称'])
    codes_data = codes_data.drop_duplicates(subset = ['name'])
    return codes_data

def read_fama_french_returns_data():
    fama_french_returns = pd.read_excel('STK_MKT_THRFACDAY.xlsx')
    fama_french_returns.columns = ['date', 'market', 'smb', 'hml']
    fama_french_returns['date'] = pd.to_datetime(fama_french_returns['date'])
    fama_french_returns = fama_french_returns.set_index('date')
    return fama_french_returns

def read_stock_returns():
    stock_returns = pd.read_csv('stock_returns_quarterly.csv')
    stock_returns['date'] = pd.to_datetime(stock_returns['date'])
    stock_returns = stock_returns.set_index(['code', 'date'])
    stock_returns.index.names = ['stock_code', 'date']
    stock_returns.columns = ['stock_returns']
    return stock_returns

def read_stock_returns_monthly():
    stock_returns = pd.read_csv('stock_returns_monthly.csv')
    stock_returns['date'] = pd.to_datetime(stock_returns['date'])
    stock_returns = stock_returns.set_index(['stock_code', 'date'])
    stock_returns.index.names = ['stock_code', 'date']
    stock_returns.columns = ['stock_returns']
    return stock_returns

def read_stock_returns_daily():
    stock_daily_price = pd.read_csv('股票日度价格数据.csv')
    stock_daily_price['date'] = pd.to_datetime(stock_daily_price['date'])
    stock_daily_price = stock_daily_price.set_index(['code', 'date'])
    stock_returns = stock_daily_price.sort_index().groupby('code').pct_change()
    stock_returns.columns = ['stock_returns']
    stock_returns.index.names = ['stock_code', 'date']
    return stock_returns

def read_fund_expense_ratio():
    data_list = []
    for filename in ['基金费率变动/Fund_FeesChange.xlsx', '基金费率变动/Fund_FeesChange1.xlsx', '基金费率变动/Fund_FeesChange2.xlsx']:
        data = pd.read_excel(filename)
        data = data.iloc[2:].dropna(subset = ['Symbol'])
        data = data[data['NameOfFee'].isin(['托管费率', '管理费率', '销售服务费率'])]
        data = data[['Symbol', 'DeclareDate', 'NameOfFee', 'ProportionOfFee', 'EffectiveDate']]
        data.columns = ['code', 'date1', 'fee_name', 'expense', 'date']
        data['code'] += '.OF'
        data['date'] = pd.to_datetime(data['date'])
        data['date1'] = pd.to_datetime(data['date1'])
        data['date'] = data['date1'].fillna(data['date'])
        data = data.set_index(['code', 'date']).sort_index()
        data['expense'] = pd.to_numeric(data['expense'], errors='coerce')
        data = data.groupby(['code', 'date'])['expense'].sum()
        data_list.append(data)
    return data

def read_holds_winners_returns():
    data = pd.read_csv('fund_holds_winners_daily_returns.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['code', 'date'])
    return data