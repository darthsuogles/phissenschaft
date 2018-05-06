"""
Essential plotting
"""
import numpy as np
import pandas as pd

def weekly_user_activity():
    weekdays = ["M", "T", "W", "Th", "F"]
    vals_map = dict(
        [(k, numpy.random.randint(low=1, high=15, size=len(weekdays)))
         for k in ["A", "B", "C", "D", "E"]])

    df = pd.DataFrame(vals_map, index=weekdays)

    ax = df.plot(kind='bar', stacked=True, grid=True, colormap='Spectral')
    ax.set_title('user activity')
    ax.set_xticklabels(weekdays, rotation='horizontal')

weekly_user_activity()    


# Generate a sequence of dates
date_rng = pd.date_range('1/1/2011', periods=16, freq='M')
n = len(date_rng)
f1 = np.random.randn(n)
f2 = np.random.randn(n)

df_14 = pd.DataFrame({'2014': f1}, index=date_rng)
df_15 = pd.DataFrame({'2015': f2}, index=date_rng)
df = pd.concat([df_14, df_15], axis=1)
# try: df_orig
# except NameError: df_orig = pd.read_csv('dsakdsahduisa.csv')
#df = pd.DataFrame({'2014': f1, '2015': f2}, index=date_rng)

ax = df.plot(kind='bar', stacked=False)
ax.set_title('yearly death comparison')
