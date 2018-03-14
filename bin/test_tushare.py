import tushare as ts
day = '0313'
concept = '../data/' + day + '_concept.csv'
industry = '../data/' + day + '_industry.csv'
index = '../data/' + day + '_index.csv'
def get_industry():
    df = ts.get_industry_classified()
    df.to_csv(industry, encoding="utf-8")

def get_concept():
    df = ts.get_concept_classified()
    df.to_csv(concept,encoding="utf-8")

def get_index():
    df = ts.get_index()
    df.to_csv(index, encoding="utf-8")

if __name__ == '__main__':
    #get_concept()
    #get_industry()
    get_index()