import jieba
import pandas as pd

def get_view_data(content_data):
    content_data = content_data.replace('\s|\\t', '', regex = True)
    view1 = content_data.str.extractall(r'简要展望((?:(?!\.\.\.|\u2026).)*?。)\d').unstack('match')
    # view1 = content_data.str.extractall(r'简要展望(.+。)\d').unstack('match')
    if view1.shape[1] > 0:
        temp = view1.iloc[:, 1]
        temp = temp.fillna(view1.iloc[:, 0])
        view1 = temp.rename('view1')
    view2 = content_data.str.extractall('展望([^(\.\.)]+)\d.').unstack('match')
    if view2.shape[1] > 0:
        temp = view2.iloc[:, 1]
        temp = temp.fillna(view2.iloc[:, 0])
        view2 = temp.rename('view2')
    view3 = content_data.str.extractall('展望([^(\.\.)]+)\d.').unstack('match')
    if view3.shape[1] > 0:
        temp = view3.iloc[:, 1]
        temp = temp.fillna(view3.iloc[:, 0])
        view3 = temp.rename('view3')
    index = content_data.index
    view = view1.reindex(index).fillna(view2.reindex(index)).fillna(view3.reindex(index))
    view4 = view.str.extract('(.+)§').iloc[:, 0]
    view = view4.fillna(view)
    return view

def print_long_text(text, line_length = 20):
    for i in range(0, len(text), 20):
        print(text[i:(i+line_length)])

def read_dictionary():
    dictionary = {}
    positive_words1 = pd.read_excel("Tones in Manager's Report/金融领域中文情绪词典.xlsx", sheet_name='年报正面',
                                    header=None).iloc[:, 0].unique()
    positive_words2 = pd.read_excel("Tones in Manager's Report/金融领域中文情绪词典.xlsx", sheet_name='社媒正面',
                                    header=None).iloc[:, 0].unique()
    negative_words1 = pd.read_excel("Tones in Manager's Report/金融领域中文情绪词典.xlsx", sheet_name='年报负面',
                                    header=None).iloc[:, 0].unique()
    negative_words2 = pd.read_excel("Tones in Manager's Report/金融领域中文情绪词典.xlsx", sheet_name='社媒负面',
                                    header=None).iloc[:, 0].unique()
    stop_words = pd.read_csv("Tones in Manager's Report/stopwords_cn.txt", sep = '\\n',
                             header = None).iloc[:, 0].unique()
    dictionary['positive'] = set(positive_words1) | set(positive_words2)
    dictionary['negative'] = set(negative_words1) | set(negative_words2)
    dictionary['stop'] = set(stop_words)
    return dictionary

def get_cut_count(view_str, dictionary):
    words_cut = jieba.cut(view_str)
    pos_count = 0
    neg_count = 0
    all_count = 0
    for word in words_cut:
        if word in dictionary['stop']:continue
        if word in dictionary['positive']:pos_count+=1
        if word in dictionary['negative']:neg_count+=1
        all_count+=1
    return pd.Series({'pos_count':pos_count, 'neg_count':neg_count, 'all_count':all_count})

def fill_tone(fund_returns_daily, tone_factor):
    tone_factor.index.names = ['code', 'date']
    used_index = fund_returns_daily.index.union(tone_factor.index).unique()
    tone_factor = tone_factor[~tone_factor.index.duplicated()]
    tone_factor = tone_factor.reindex(used_index).sort_index()
    tone_factor = tone_factor.groupby('code').ffill()
    return tone_factor

def tone_group_returns(fund_nav_daily, tone_factor, freq = 'QE'):
    combined_data = pd.concat([fund_nav_daily, tone_factor], axis = 1)
    dates = combined_data.index.get_level_values('date').unique()
    dates = pd.Series(dates, index = dates)
    used_dates = dates.sort_index().resample(freq).last()
    sub_data = combined_data[combined_data.index.get_level_values('date').isin(used_dates)]
    sub_data['group'] = combined_data['tones_factor'].groupby('date').rank(pct = True) * 100 // 20 + 1
    sub_data['group'] = sub_data['group'].mask(sub_data['group'] == 6, 5)
    sub_data = sub_data.sort_index()
    sub_data['returns'] = sub_data['f_nav_adjusted'].groupby('code').pct_change()
    sub_data['returns_next'] = sub_data['returns'].groupby('code').shift(-1)
    group_returns = sub_data.groupby(['group', 'date'])['returns_next'].mean().unstack('group').sort_index()
    return group_returns

