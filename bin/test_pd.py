import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../data/train.csv', header=0)
print(type(df))
print(df.head(3))
print(df.dtypes)
print(df.info())
print(df.describe())
print(df['Age'][0:10])
print(type(df['Age']))
print(df[df['Age']>60])
print(df[df['Age']>60][['Sex','Pclass','Age','Survived']])
print(df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']])

for i in range(1,4):
    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))

import pylab as P
df['Age'].hist()
P.show()

df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
print(median_ages)

df['AgeFill'] = df['Age']
print(df.head())
print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10))

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)  )