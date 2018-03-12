#encoding=utf8
import tushare as ts
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import sklearn
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.utils import shuffle

from pypinyin import lazy_pinyin, Style
import pypinyin

#predictors = "mktcap", "nmc", "industry", "area", "pe", "pb_y", "timeToMarket","rev","profit","npr","holders"]
filters = "mktcap|nmc|industry_*|area_*|pe|pb_y|timeToMarket|rev|profit|npr|holders"
day = '0312'
basic = '../data/' + day + '.csv'
info = '../data/' + day + '_info.csv'

def get_basics():
    df = ts.get_stock_basics()
    df.to_csv(basic,encoding="utf-8")

def get_marketinfo():
    df = ts.get_today_all()
    df.to_csv(info,encoding="utf-8")

def merge():
    target = pd.read_csv(basic)

    attr = pd.read_csv(info)
    df = pd.merge(target, attr, on='code')
    return df

    #print(df[['name_x','zd']])
    #print(df.dtypes)
    #print(df.info)

def preprocess(df):
    df = df[(df.changepercent >= 1.0)|(df.changepercent <= -1.0)]
    df['zd'] = df['changepercent'].map(lambda x: 1 if x > 0 else 0)

    df["area"] = df["area"].fillna('其它')
    df['area'] = df['area'].map(lambda x: '_'.join(lazy_pinyin(x, style=Style.TONE2)))
    df['industry'] = df['industry'].map(lambda x: '_'.join(lazy_pinyin(x, style=Style.TONE2)))

    dummies_Industry = pd.get_dummies(df['industry'], prefix='industry')
    dummies_Area = pd.get_dummies(df['area'], prefix='area')
    df = pd.concat([df, dummies_Industry,dummies_Area], axis=1)


    '''
    le = sklearn.preprocessing.LabelEncoder()
    # 非数值型转化为数值型
    le.fit(df['industry'])

    df['industry'] = le.transform(df['industry'])

    df["area"] = df["area"].fillna('其它')

    le.fit(df['area'])
    df['area'] = le.transform(df['area'])
    '''

    df.drop(['code','changepercent','trade','open','high','low','area','industry','settlement','name_y','per','pb_x'], axis=1, inplace=True)

    return df
    #print(dummies_Area)

def grid_search_train(df):
    df = shuffle(df)
    train_df = df.filter(regex=filters)
    results = []
    sample_leaf_options = list(range(1, 10, 1))
    n_estimators_options = list(range(1, 30, 1))
    n_depth_options = list(range(1,10,1))
    groud_truth = df['zd'][801:]


    for leaf_size in sample_leaf_options:
        for n_estimators_size in n_estimators_options:
            for n_depth_size in n_depth_options:
                alg = RandomForestClassifier(max_depth=n_depth_size,min_samples_leaf=leaf_size, n_estimators=n_estimators_size,criterion='entropy')
                alg.fit(train_df[:801],df['zd'][:801])
                predict = alg.predict(train_df[801:])
                # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
                results.append((n_depth_size,leaf_size, n_estimators_size, (groud_truth == predict).mean()))
                #  真实结果和预测结果进行比较，计算准确率
                print((groud_truth == predict).mean())

    # 打印精度最大的那一个三元组
    print(max(results, key=lambda x: x[3]))

def train(df):
    # 用正则取出我们要的属性值
    #train_df = df.filter(regex='zd|mktcap|nmc|pe|outstanding|totals|totalAssets|liquidAssets|fixedAssets|reserved|reservedPerShare|esp|bvps|pb_y|timeToMarket|undp|perundp|rev|profit|gpr|npr|holders|area_.*|industry_.*')
    #train_df = df.filter(regex='zd|mktcap|nmc|industry|area|pe|outstanding|totals|totalAssets|liquidAssets|fixedAssets|reserved|reservedPerShare|esp|bvps|pb_y|timeToMarket|undp|perundp|rev|profit|gpr|npr|holders')
    train_df = df.filter(regex=filters)
    #print(df[(df.holders<=30522.5)&(df.profit>443.77)][['name_x','zd']])
    #train_np = df[predictors].as_matrix()

    print(df['zd'].value_counts())
    '''
    # y即zd结果
    print(y)
    y = train_np[:, 0]
    # X即特征属性值
    X = train_np[:, 1:]
    '''

    X = train_df
    y = df['zd']

    i = 0
    for col in X.columns:
        print(i,col)
        i += 1

    #print(X.shape)
    #print(y.shape)

    #gbm0 = GradientBoostingClassifier(random_state=10)

    gbm0 = RandomForestClassifier(n_estimators=6,criterion='gini',max_depth=5,min_samples_leaf=8)


    gbm0.fit(X, y)
    print(gbm0.feature_importances_)
    print(gbm0.classes_)
    y_pred = gbm0.predict(X)
    y_predprob = gbm0.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
    importances = gbm0.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        #if f >= 50:break
        print("%d. feature %d(%s) (%f)" % (f + 1, indices[f],X.columns[indices[f]], importances[indices[f]]))


    numTrees = len(gbm0.estimators_)
    for num in range(0,numTrees):
        dot_data = StringIO()
        tree.export_graphviz(gbm0.estimators_[num].tree_,feature_names=X.columns,class_names=["descend","ascend"], out_file=dot_data)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("../result/" + day + "_" + str(num) + "_trees.pdf")

        #tree.export_graphviz(gbm0.estimators_[num].tree_,out_file = str(num) + 'tree.dot')
    '''
    gbm0 = tree.DecisionTreeClassifier()
    gbm0.fit(X, y)
    print(gbm0.feature_importances_)
    print(gbm0.classes_)
    y_pred = gbm0.predict(X)
    y_predprob = gbm0.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
    importances = gbm0.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        # if f >= 50:break
        print("%d. feature %d(%s) (%f)" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))

    dot_data = tree.export_graphviz(gbm0,feature_names=X.columns,class_names=["descend","ascend"], out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")
    '''

    print(cross_validation.cross_val_score(gbm0, X, y, cv=5))


if __name__ == '__main__':
    get_basics()
    get_marketinfo()
    df = merge()
    df = preprocess(df)
    #grid_search_train(df)
    train(df)


    '''
    c = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C']
    category = pd.Categorical(c)
    # 接下来查看category的label即可
    print(category.labels)
    '''