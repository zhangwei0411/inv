import tushare as ts
day = '0313'
info = '../data/' + day + '_concept.csv'
#print(ts.get_industry_classified())
df = ts.get_concept_classified()
df.to_csv(info,encoding="utf-8")