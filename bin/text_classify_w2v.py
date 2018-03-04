# coding=utf-8
import sys,re
from operator import itemgetter


from xml.dom.minidom import parse
import xml.dom.minidom
from bs4 import BeautifulSoup

import gensim
from gensim import corpora, models, similarities
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np

import jieba
import jieba.posseg
import jieba.analyse

#sys.path.append("jieba.zip/jieba")
class word2vec:
    def gen_train_test_sample(self,pos_path,neg_path):
        with open(pos_path, 'r') as infile:
            self.pos_samples = infile.readlines()
        with open(neg_path, 'r') as infile:
            self.neg_samples = infile.readlines()
        y = np.concatenate((np.ones(len(self.pos_samples)), np.zeros(len(self.neg_samples))))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.concatenate((self.pos_samples, self.neg_samples)), y, test_size=0.2)

    def gen_train_sample(self,path):
        with open(path, 'r',encoding='utf-8') as infile:
            self.x_train = infile.readlines()


    def preprocess(self):
        self.x_train = [z.lower().split('\t',1)[1].replace('\n','').split() for z in self.x_train if len(z) > 10 ]


    def build_w2v(self,n_dim,cnt):
        self.imdb_w2v = Word2Vec(size=n_dim, min_count=cnt)
        self.imdb_w2v.build_vocab(self.x_train)
        self.imdb_w2v.train(self.x_train,total_examples=self.imdb_w2v.corpus_count,epochs=self.imdb_w2v.iter)

    def getWordVec(self,text,size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += self.imdb_w2v[word].reshape((1, size))
                #print(word)
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def train_model(self):
        self.train_vecs = np.concatenate([self.getWordVec(z, 300) for z in self.x_train])
        self.lr = SGDClassifier(loss='log', penalty='l1')
        self.lr.fit(self.train_vecs, self.y_train)
        self.test_vecs = np.concatenate([self.getWordVec(z, 300) for z in self.x_test])
        print('Test Accuracy: %.2f'%self.lr.score(self.test_vecs, self.y_test))

    def load_w2v(self,name):
        self.imdb_w2v = Word2Vec.load(name) 
        return self.imdb_w2v

    def save_w2v(self,name):
        self.imdb_w2v.save(name)

    def load_model(self,name):
        self.lr = joblib.load(name)

    def save_model(self,name):
        joblib.dump(self.lr, name)

    def predict_label(self,content):
        result = jieba.cut(content)
        s = ' '.join(result)
        print(self.lr.predict(self.getWordVec(s.lower().replace('\n','').split(), 300)),self.lr.predict_proba(self.getWordVec(s.lower().replace('\n','').split(), 300)))

        
    def dump_vec(self,path):
        f = open('c4.txt', 'w', encoding='utf-8')
        with open(path, 'r',encoding='utf-8') as infile:
            self.x_train = infile.readlines()
        for s in self.x_train:
            if len(s.strip()) == 0:continue
            f.write((s.lower().split('\t',1)[0].replace(',',' ') + ',' + ','.join( str(f) for f in self.getWordVec(s.lower().split('\t',1)[1].replace('\n','').split(), 100)[0])) + "\n")
        f.close()

if __name__ == '__main__':


    w2v = word2vec()

    w2v.gen_train_sample('clean_corpus.txt')
    w2v.preprocess()
    w2v.build_w2v(100,2)
    w2v.save_w2v('c_w2v.m')

    m = w2v.load_w2v('c_w2v.m')
    w2v.dump_vec('clean_corpus.txt')
    '''
    print('------')
    for k,v in m.similar_by_word('油耗',50):
        print k.encode('utf-8'),v
    
    print('---上市---')
    for k,v in m.similar_by_word('上市',topn=50):
        print k.encode('utf-8'),v
    print('---促销---')
    for k,v in m.similar_by_word('促销',topn=50):
        print k.encode('utf-8'),v
    print('---试驾---')
    for k,v in m.similar_by_word('试驾'):
        print k.encode('utf-8'),v
    print('---动力---')
    for k,v in m.similar_by_word('动力'):
        print k.encode('utf-8'),v
    print('---油耗---')
    for k,v in m.similar_by_word('油耗'):
        print k.encode('utf-8'),v
    print('---安全---')
    for k,v in m.similar_by_word('安全'):
        print k.encode('utf-8'),v
    print('---操控---')
    for k,v in m.similar_by_word('操控'):
        print k.encode('utf-8'),v    
    print('---内饰---')
    for k,v in m.similar_by_word('内饰'):
        print k.encode('utf-8'),v    
    print('---灯光---')
    for k,v in m.similar_by_word('灯光'):
        print k.encode('utf-8'),v    
    '''
