"""
    ## Invoices ##

    * created_at (when invoice was created in the system)
    * due (the last date day invoice should be paid)
    * last_payment_date (actual invoces payment date)
    * total (invoice sum)
    * paid (sum that is already paid by this invoice)
    * source_system (how invoices extracted)

    ## Customers ##

    * created_at (first time customer appeared in the system)
    * score (automatically calculated by system customer score, lower -- better)


    Your task is to conduct exploratory data analysis and build a model that will predict
    whether the invoice will be paid in time or will be overdue.
    In case of overdue, your model/approach should provide the info when the invoice should be paid.
    It can be either regression or classification (with periods chosen up to you).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




invoices = pd.read_csv('data/invoices.csv').sort_values(['created_at'], ascending=1).set_index('invoice_id')
customers = pd.read_csv('data/customers.csv').set_index('customer_id')


cols = [col for col in invoices if col not in ['exchange_rate', 'source_system', 'currency']]
full_data = invoices[cols].merge(customers, left_on='customer_id', right_index=True, how='left', suffixes=['', '_cust'])

for col in ['created_at', 'due', 'last_payment_date', 'created_at_cust']:
    full_data[col] = pd.to_datetime(full_data[col], errors='coerce')

full_data['days_creat_due'] = (full_data['created_at'] - full_data['due']).astype('timedelta64[D]')
full_data['days_last_due'] = (full_data['last_payment_date'] - full_data['due']).astype('timedelta64[D]')


def paid_st_agg(g):
    # 0 - no UNPADEs
    # 1 - PAID after UNPAID
    # 2 - All UNPADEs

    if g[g == 'UNPAID'].empty:
        return 0
    elif g[g != 'UNPAID'].empty:
        return 2
    else:
        return int(g.index.get_loc(g[g == 'UNPAID'].index.min()) < g.index.get_loc(g[g != 'UNPAID'].index.max()))


cov = full_data.groupby('customer_id').agg(
    {'days_creat_due': 'mean',
     'days_last_due': lambda g: g.dropna().mean(),
     'score': 'max',
     'paid_status': paid_st_agg})

cov['score_norm'] = (cov['score'] - cov['score'].min()) / (cov['score'].max() - cov['score'].min())

# a = cov.plot.scatter(x='score',y='paid_status')
# plt.show()


g = full_data.groupby('customer_id')
for i in list(g)[:10]:
    print(i)

l = list(g)
mult_unpaid = [i for i in l if i[0]==1850911]
# l[2][1].sort_values(['created_at'], ascending=1)

# covariance_moment
cov[['score', 'days_last_due']].cov().iloc[1, 0] / (np.sqrt(cov[['score', 'days_last_due']].cov().iloc[0, 0]) * np.sqrt(cov[['score', 'days_last_due']].cov().iloc[1, 1]))


