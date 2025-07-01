import pandas as pd

def display_regress_result(model):
    base_table = model.summary2().tables[1]
    base_table = base_table[['Coef.', 'P>|t|']]
    base_table.columns = ['coef', 'p']
    base_table['coef'] = base_table['coef'].apply(lambda x:"{:.4f}".format(x))
    base_table['stars'] = ''
    base_table['stars'] = base_table['stars'].mask(base_table['p'] < 0.1, '*')
    base_table['stars'] = base_table['stars'].mask(base_table['p'] < 0.05, '**')
    base_table['stars'] = base_table['stars'].mask(base_table['p'] < 0.01, '***')
    base_table['coef'] += base_table['stars']
    base_table['coef'] += ('\n' + base_table['p'].apply(lambda x:"{:.3f}".format(x)))
    result = base_table['coef']
    result['Observations'] = "{:,}".format(int(model.nobs))
    result['Adj R Square'] = "{:.4f}".format(model.rsquared_adj)
    return result
