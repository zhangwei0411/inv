#encoding=utf8
import tushare as ts
import jieba
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn.preprocessing import scale

def load_csv(name):
    df = pd.read_csv(name)

    f = open('tmp.txt', 'w',encoding='utf-8')
    for i in range(len(df)):
        f.write(str(df["title"][i]).strip() + "\t" + str(df["content"][i]).replace('\r\n','').strip() + "\r\n")
    f.close()


def cleanText(corpus):
    f = open('clean_corpus.txt', 'w',encoding='utf-8')
    corpus = [(line.split('\t',1)[0], ' '.join(jieba.cut(line))) for line in corpus]
    for z in corpus:
        f.write(z[0].strip() + "\t" + z[1].lower().replace('\n','') + "\r\n")
    f.close()


def get_latest_news():
    df = ts.get_latest_news(top=1000,show_content=True)
    df.to_csv("news_1000_0314.csv",encoding="utf-8")



if __name__ == '__main__':
    #get_latest_news()

    load_csv("news_1000_0314.csv")
    with open('tmp.txt', 'r',encoding='utf-8') as infile:
        news_tweets = infile.readlines()

    x_train = cleanText(news_tweets)



