# %%
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import os
from dotenv import load_dotenv

# %%
load_dotenv()
#os.environ['OpenAI_API_Key'] = os.getenv('OpenAI_API_Key')

# %%
#os.getenv('OpenAI_API_Key')

# %%
llm = OpenAI(api_token=os.getenv('OpenAI_API_Key'))

# %%
#os.getenv('filepath')

# %%
df1=pd.read_csv(os.getenv('filepath'))
df1.head()

# %%
df2=pd.read_csv(os.getenv('sales_filepath'),encoding='unicode_escape')
df2.head()

# %%
pandas_ai = PandasAI(llm, conversational=False)
pandas_ai(df2, prompt='Which are the top 5 City with highest sales and what are their sales value?')

# %%
pandas_ai = PandasAI(llm, conversational=False)
pandas_ai(df1, prompt='Which department has highest employee churn?what is its churn percentage')

# %%
pandas_ai(df1, prompt='What is the total number of employee?')

# %%
pandas_ai(df1, prompt='What is the total number of employee in finance department with status left?')

# %%
pandas_ai(df1, prompt='What is the total number of employee in finance department?')

# %%
195/728

# %%
pandas_ai(df1, prompt='Create bar plot for top 5 deparment with highest churn rate along with displaying data')

# %%
df1[(df1['status']=='Left') & (df1['department']=='sales')].shape[0]

# %%
def churn_rate(department):
    num = df1[(df1['status']=='Left') & (df1['department']==department)].shape[0]
    den = df1[df1['department']==department].shape[0]
    if den == 0:
        churn = "dep na"
    else:
        churn = num/den
    return churn

# %%
dep = df1.department.unique().tolist()
dep[0], len(dep)

# %%
for i in range(0,len(dep)-1):
    print(dep[i],churn_rate(dep[i]))


