import pandas as pd
import statsmodels.formula.api as smf
from . import configs

'''
Running test regressions 
'''

data=pd.read_csv(configs.data_dir + 'compiled_measures_lagged' + str(configs.lag) + '.csv')

model=smf.ols(formula='total_citations ~ application_novelty + C(grant_year) + C(cpc_4)', data=data)
result=model.fit()
result_summary=result.summary()
print(result_summary)

model=smf.poisson(formula='xi_real ~ application_novelty +  C(grant_year) + C(cpc_4)', data=data)
result=model.fit()
result_summary=result.summary()
print(result_summary)
