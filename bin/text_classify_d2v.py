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
from gensim.models import Doc2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import jieba
import jieba.posseg
import jieba.analyse
LabeledSentence = gensim.models.doc2vec.LabeledSentence
class doc2vec:
    def gen_train_test_sample(self,pos_path,neg_path,dm):
        with open(pos_path, 'r',encoding='utf-8') as infile:
            self.pos_samples = infile.readlines()
            dm_vec = [dm.infer_vector(jieba.cut(line.strip())) for line in self.pos_samples]
            #dbow_vec = [dbow.infer_vector(jieba.cut(line.strip())) for line in self.pos_samples]
            #self.pos_samples = np.hstack((np.array(dm_vec), np.array(dbow_vec)))
            self.pos_samples = dm_vec
        with open(neg_path, 'r',encoding='utf-8') as infile:
            self.neg_samples = infile.readlines()
            dm_vec = [dm.infer_vector(jieba.cut(line.strip())) for line in self.neg_samples]
            #dbow_vec = [dbow.infer_vector(jieba.cut(line.strip())) for line in self.neg_samples]
            #self.neg_samples = np.hstack((np.array(dm_vec), np.array(dbow_vec)))
            self.neg_samples = dm_vec
        y = np.concatenate((np.ones(len(self.pos_samples)), np.zeros(len(self.neg_samples))))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.concatenate((self.pos_samples, self.neg_samples)), y, test_size=0.2)

    def gen_train_sample(self,path):
        with open(path, 'r',encoding='utf-8') as infile:
            self.x_train = infile.readlines()


    def preprocess(self):
        self.corpus = [z.lower().split('\t', 1)[0].replace(',',' ') for z in self.x_train if len(z) > 10]
        self.x_train = [z.lower().split('\t',1)[1].replace('\n','').split() for z in self.x_train if len(z) > 10 ]


    # Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    # We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    # a dummy index of the review.
    def labelizeReviews(self,reviews, label_type):
        labelized = []
        for i, v in enumerate(reviews):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized


    def build_d2v(self,size,cnt):
        self.x_train = self.labelizeReviews(self.x_train, 'TRAIN')

        '''
        # instantiate our DM and DBOW models
        self.model_dm = gensim.models.Doc2Vec(min_count=cnt, window=10, size=size, sample=1e-3, negative=5, workers=3)
        self.model_dbow = gensim.models.Doc2Vec(min_count=cnt, window=10, size=size, sample=1e-3, negative=5,dm=0, workers=3)

        # build vocab over all reviews
        self.model_dm.build_vocab(self.x_train)
        self.model_dbow.build_vocab(self.x_train)

        # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        all_train_reviews = self.x_train

        for epoch in range(10):
            perm = np.random.permutation(len(all_train_reviews))
            self.model_dm.train(all_train_reviews,total_examples=self.model_dm.corpus_count,epochs=self.model_dm.iter)
            self.model_dbow.train(all_train_reviews,total_examples=self.model_dbow.corpus_count,epochs=self.model_dbow.iter)
        '''
        self.model_dm = gensim.models.Doc2Vec(min_count=cnt, window=5, size=size, sample=1e-3, negative=5, workers=3,hs=1,iter=6)
        self.model_dm.train(self.x_train, total_examples=self.model_dm.corpus_count, epochs=70)

    # Get training set vectors from our models
    def getVecs(self,model,corpus, size):
        train_arrays = np.zeros((len(corpus), size))
        for i,v in enumerate(corpus):
            train_arrays[i] = model.docvecs["TRAIN_%s" % i]
        return train_arrays

    def text_classify(self):
        from sklearn.linear_model import SGDClassifier

        lr = SGDClassifier(loss='log', penalty='l1')
        lr.fit(self.x_train, self.y_train)

        print('Test Accuracy: %.2f' % lr.score(self.x_train, self.y_train))

        pred_probas = lr.predict_proba(self.x_train)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_train, pred_probas)
        roc_auc = auc(fpr, tpr)
        print('area = %.2f' %roc_auc)

        plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')

        plt.show()


    def dump_vec(self,corpus,train_vecs):
        f = open('c4.txt', 'w', encoding='utf-8')
        for i,v in enumerate(corpus):
            f.write(v + ',' + ','.join( str(f) for f in train_vecs[i]) + "\n")
        f.close()

    def load_d2v_dm(self,name):
        self.model_dm = Doc2Vec.load(name)
        return self.model_dm

    def load_d2v_dbow(self,name):
        self.model_dbow = Doc2Vec.load(name)
        return self.model_dbow


    def save_d2v_dm(self,name):
        self.model_dm.save(name)

    def save_d2v_dbow(self, name):
        self.model_dbow.save(name)

    def text_cluster(self,corpus,train_vecs,num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans_model = kmeans.fit(np.array(train_vecs))
        print(metrics.silhouette_score(np.array(train_vecs), kmeans_model.labels_, metric='euclidean'))

if __name__ == '__main__':

    d2v = doc2vec()



    '''
    d2v.gen_train_sample('clean_corpus.txt')
    d2v.preprocess()
    '''



    #d2v.build_d2v(200,1)
    #d2v.save_d2v_dm("d2v_dm.m")
    #d2v.save_d2v_dbow("d2v_dbow.m")





    d2v.load_d2v_dm("d2v_dm.m")
    #d2v.load_d2v_dbow("d2v_dbow.m")
    d2v.gen_train_test_sample("positive.txt", "negative.txt", d2v.model_dm)
    d2v.text_classify()


    #train_vecs_dm = d2v.getVecs(d2v.model_dm,d2v.x_train, 100)

    #train_vecs_dbow = d2v.getVecs(d2v.model_dbow,d2v.x_train,100）


    '''
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    d2v.dump_vec(d2v.corpus,train_vecs)
    for i in range(2,20):
        d2v.text_cluster(d2v.corpus,train_vecs,i)
    print(train_vecs.shape)
    '''


    inferred_vector = d2v.model_dm.infer_vector(jieba.cut("新华社 评雄安 新区 建设 一周年 : 建 千秋 之 城   办 国家 大事 _ 03 - 31   17 : 53 	 新华社 石家庄 3 月 31 日电   题 ： 建 千秋 之 城   办 国家 大事 — — 河北 雄安 新区 建设 发展 一周年 纪实 新华社 记者 孙杰 、 王 洪峰 、 郁琼源 淀阔 水清 凭 鱼跃 ， 春暖苇 绿任 鸟飞 。 白洋淀 ， 承载 着 厚重 的 历史 ， 见证 着 雄安 新区 春天 的 故事 。 自 2017 年 4 月 1 日 横空出世 ， 白洋淀 所在 的 河北 雄安 新区 设立 至今 已届 周年 。 这片 热土 发生巨变 ： “ 千年 秀林 ” 工程 稳步 推进 ， 人造 森林 茁壮成长 ； “ 雄安 城建 第一 标 ” — — 市民 服务中心 基本 建成 ； 京雄 城际 铁路 顺利 开工 ； 100 多家 高端 高新 企业 成功 落户 … … 一年 来 ， 雄安 新区 坚持 “ 世界 眼光 、 国际标准 、 中国 特色 、 高点 定位 ” ， 贯彻 新 发展 理念 ， 努力 打造 创新 发展 示范区 。 这座 未来 之 城 ， 正 迈出 坚实 有力 的 步伐 。 有 质量 才 有 生命 — — 创造 “ 雄安 质量 ” 春到 雄安 ， 生机盎然 。 雄县 昝岗镇 赵岗村 ， 一幅 热火朝天 的 场景 ： 重型 卡车 满载 树苗 驶来 ， 工人 们 紧张 而 有序 地 将 一棵 棵 侧柏 、 国槐 、 油松 等 植入 坑内 … … 这是 雄安 新区 “ 千年 秀林 ” 工程 的 施工现场 。 每棵 苗木 上 都 挂 着 一个 带有 二维码 的 铝制 小 标牌 。 “ 这是 它们 的 专属 身份证 。 ” 中国 雄安 集团 生态 公司 专家 徐 成立 博士 说 ， “ 这个 二维码 链接 着 雄安 森林 大 数据系统 ， 通过 它 可 查询 苗木 的 树种 、 规格 、 位置 、 生长 信息 等 ， 实现 对 苗木 全 生命 过程 监控 。 ” 高起点 ， 新 梦想 。 雄安 新区 坚持 生态 优先 ， 建设 一座 绿色生态 之 城 。 “ 千年 秀林 ” 是 雄安 新区 坚持 “ 生态 优先 、 绿色 发展 ” 理念 的 具体 体现 。 去年 秋季 ， 已经 植树 26 万株 ， 今年 将 再造 10 万亩 苗景 兼用 林 ， 未来 将 达到 百万亩 。 同时 ， 白洋淀 生态 修复 工程 紧锣密鼓 进行 。 2017 年 共 整治 垃圾 86 万立方米 ， 淀区 内 45 个 乡村 生活 垃圾 实现 统一 收集 转运 ， 修建 147 个 小型 污水处理 站 ， 完成 生态 补水 近 8000 万立方米 ， 生态环境 稳步 改善 。 “ 为雄安 新区 打 好 蓝绿 交织 的 底色 ， 才能 配得 上 这座 未来 之 城 、 千秋 之 城 ！ ” 徐 成立 说 。 展现 “ 雄安 质量 ” 的 ， 还有 被誉为 “ 雄安 城建 第一 标 ” 的 市民 服务中心 — — 2017 年 12 月 7 日 正式 开工 ， 2018 年 3 月 28 日 基本 建成 ， 占地 1100 亩 ， 建筑面积 超过 10 万平方米 ， 比同 体量 工程施工 速度 快 2 - 3 倍 。 高速度 建设 背后 是 高技术 的 支撑 。 “ 这里 使用 了 30 多项 建筑 新 技术 ， 在 国内 民用 建筑史 上 尚属 首次 。 ” 市民 服务中心 项目 施工 联合体 技术 总监 叶建 介绍 ， 8 栋 单体 全部 为 装配式 建筑 ， 每个 构件 都 “ 埋 ” 有 芯片 或 二维码 ； 项目 以大 数据中心 为 枢纽 ， 上线 了 智慧 建造 系统 ， 只 需 使用 电脑 或 手机 就 可以 实现 全景 监控 、 环境 能耗 监测 、 无人机 航拍 等 功能 。 中国 雄安 集团 总经理 助理 杨忠 说 ， 雄安 市民 服务中心 相当于 一个 小型 城市 的 缩影 ， 是 新区 功能定位 与 发展 理念 的 率先 呈现 ， 目前 运用 的 这些 新 技术 ， 将 在 今后 新区 的 整体 建设 中 大面积 推广 ， 也 为 我国 民用建筑 提供 范例 。 2 月 28 日 ， 新区 首个 重大 交通 项目 京雄 城际 铁路 开工 建设 。 “ 这 将 为 新区 集中 承接 北京 非 首都 功能 疏解 提供 有力 支撑 ， 对 促进 京津冀 协同 发展 具有 重要 作用 。 ” 京雄 城际 铁路 雄安 建设 指挥部 指挥 长 杨斌 介绍 ， 项目 全线 建成 后 ， 从 北京 城区 到达 雄安 新区 仅 需 30 分钟 左右 。 蓝色 之 城 、 绿色 之 城 、 数字 之 城 、 现代 之 城 渐次 展开 ， 不断 为雄安 新区 “ 质量 之 城 ” 增加 注解 。 改革开放 初期 ， 深圳特区 创造 了 “ 深圳 速度 ” ； 时隔 40 年 ， 河北省委 省政府 志在 新 时代 创造 “ 雄安 质量 ” 。 有 创新 才 有 未来 — — 打造 “ 雄安 样板 ” 承接 北京 非 首都 功能 疏解 是 设立 新区 的 首要任务 ， 重点 是 要紧 跟 世界 发展 潮流 ， 有 针对性 地 培育 和 发展 科技 创新 企业 ， 发展 高端 高新产业 ， 打造 在 全国 具有 重要 意义 的 创新 驱动 发展 新 引擎 。 2017 年 12 月 20 日 ， 7 台 百度 apollo 自动 驾驶 车辆 ， 在 雄安 新区 进行 了 载人 路测 。 当日 ， 新区 与 百度 公司 签署 战略 合作 协议 ， 双方 将 在 智能 出行 、 云 基础设施 等 多个 领域 展开 深度 合作 ， 共同 将 新区 打造 为 智能 城市 新 标杆 。 在 科技 创新 驱动 引领 下 ， 华讯 方舟 集团 将 与 新区 建立 科技 创新 研究院 ， 打造 攻克 太 赫兹 科技 这一 尖端科技 的 桥头堡 。 华讯 方舟 集团 董事长 吴光胜 表示 ： “ 大批 高新 高端 产业 落地 雄安是 新 时代 下 中国 科技 与 雄安 建设 的 深度 触碰 。 ” 自雄安 新区 设立 以来 ， 中国移动 围绕 打造 智能 新区 发力 ， 在 服务 数字 雄安 、 智慧 雄安 的 进程 中 创新 新 技术 发展 业态 ， 相继 完成 了 5g 试点 、 千兆 光 宽带 及 nb - iot 物 联网 业务 上线 。 新区 积极 吸纳 和 集聚 创新 要素 资源 ， 已有 上 百家 高端 高新 企业 核准 工商 注册 登记 。 站 在 新 的 历史 起点 ， 着眼 京津冀 协同 发展 大局 ， 雄安 新区 加大 与 北京 、 天津 、 石家庄 、 保定 等 城市 融合 发展 力度 ， 并 与 京津 两市 分别 签署 战略 合作 协议 ； 雄安 新区 中关村 科技园 建设 正 大力 推进 。 “ 雄州 雾列 ， 俊采 星驰 。 ” 高端 高新产业 、 尖端科技 、 高端 人才 纷至沓来 ， 多层次 、 全 覆盖 、 人性化 的 基本 公共服务 网络 也 在 新区 有序 展开 。 雄安 新区 公共 服务局 副局长 徐志芳 介绍 ， 2017 年 8 月 17 日 ， 北京市 和 河北省 共同 签署 战略 合作 协议 ， 从 新区 最 迫切 需求 入手 ， 积极 推动 优质 教育 、 医疗卫生 等 公共服务 资源 向 新区 布局 发展 。 “ 原来 我们 是 到处 请人 ， 留 一个 名牌 大学生 很 困难 ， 新区 设立 后 ， 我们 竟 收到 了 北京 一些 科教 机构 研究员 的 求职信 。 ” 雄县 教育局 局长 赵勇鸿说 ， “ 这 在 以前 想 都 不敢 想 。 ” 千年 大计 、 教育 先行 。 不久前 ， 北京市 朝阳区 实验 小学 雄安 校区 、 北京市 第八十 中学 雄安 校区 、 北京市 六一 幼儿园 雄安 校区 、 北京市 海淀区 中关村 第三 小学 雄安 校区 挂牌 成立 。 北京市教委 与 雄安 新区 管委会 签署 了 合作 协议 ， 双方 将 在 教育 战略规划 、 重点 学校 援建 项目 、 干部 教师队伍 建设 、 推动 学校 对口 帮扶 、 强化 教育 科研 力量 等 方面 开展 深度 合作 。 河北省 正 认真落实 中央 精神 部署 ， 扎实 践行 习近平 新 时代 中国 特色 社会主义 思想 ， 有效 贯彻 高质量 发展 要求 ， 创造 “ 雄安 质量 ” ， 将雄安 新区 打造 为 全国 的 一个 样板 。 凝心 聚力筑 千秋 之 城 — — 夯实 “ 雄安 根基 ” 面对 雄安 新区 这座 千秋 之 城 、 未来 之 城 、 典范 之 城 ， 党政干部 与 广大群众 凝心 聚力 ， 共创 伟业 。 位于 容城 的 奥威 大厦 作为 雄安 新区 党政机关 的 临时 办公地 ， 一年 来 经常 彻夜 灯火通明 。 新区 持续 深化 财政 、 金融 、 科技 管理 、 “ 房地 人 ” 管理 等 体制改革 ， 创新 和 完善 雄安 特色 的 城市 建设 管理 运营 和 社会 治理 机制 。 企业主 和 普通百姓 都 主动 投身 到 新区 改革 发展 的 浪潮 中 。 3 月 18 日 ， 京雄 城际 铁路 征地 工作 启动 ， 雄县 米 家务 镇仅用 一天 时间 就 完成 了 89 亩 耕地 的 临时 征地 工作 ， 所 涉及 的 板 西村 53 户和板 东村 28 户 全都 签订 了 协议书 。 当地 干部 说 ， 虽然 故土 难舍 ， 但 为了 新区 未来 ， 百姓 识大体 、 顾大局 ， 征地 不但 速度 快 ， 而且 实现 了 零 纠纷 、 零 问题 遗留 。 容城县 以 服装 产业 闻名 ， 但 产业 层次 较 低 ， 面临 整体 外迁 。 做 了 20 多年 服装 生意 的 周艳成 说 ： “ 一方面 ， 新区 支持 我们 转型 升级 ， 有央 企垫 资助 力 我们 外迁 ； 另一方面 ， 容城 11 家 企业 抱团 发展 ， 成立 了 新 公司 ， 把 服装 产业 发展 成为 时尚 朝阳产业 。 ” 一年 来 ， 雄安 新区 急 群众 之所急 ， 开展 “ 聚力 雄安送 培训 ” 系列 活动 ， 组织 “ 企业家 大讲堂 ” ， 现已 培训 2 万余 人 ， 开发 就业 岗位 1.1 万个 ， 安置 就业 7000 余人 。 改革 创新 离不开 稳定 的 社会 环境 。 雄安 新区 严格 科学 管控 ， 严禁 违规 建设 ， 避免 社会 投资 借机 炒作 、 抬高 建设 成本 。 一年 来 ， 新区 以 “ 五项 冻结 ” “ 七个 严控 ” 为 抓手 ， 坚决 管住 “ 人 、 地 、 房 ” ， 依法 收回 待 开发 土地 520 多平方公里 ， 经受 住 了 炒房 、 炒地 、 炒 户籍 、 炒 房租 等 多重 考验 。 安 新 县委书记 杨宝昌 说 ， 我们 经受 住 了 考验 ， 稳住 了 人心 ， 最大 限度 减少 由于 管控 给 群众 生产 生活 带来 的 影响 ， 为 群众 解 难题 、 办实事 。 “ 县里 开通 了 领导 干部 电话 ‘ 直通车 ’ ， 做好 群众 短信 反映 问题 的 办理 工作 ， 各 驻村 工作组 累计 为 群众 办实事 1000 多件 、 帮扶 困难群众 1300 多户 。 ” 去年 11 月 下旬 起 ， 新区 集中 开展 “ 全面 从严治党 、 切实 转变 作风 、 密切联系 群众 ” 三项 工程 ， 深入开展 “ 大接访 、 大 走访 、 大下访 ” 活动 ， 妥善解决 了 群众 的 住房 、 就业 、 就医 、 冬季 取暖 等 现实 问题 。 随着 新区 展现 出新 的 风姿 ， 当地 群众 言语 间 都 流露出 自豪感 。 雄县 昝北村 69 岁 村民 王树信 说 ： “ 地 更 绿 、 天 更 蓝 、 水 更 清 ， 我们 的 家乡 会 越来越 好 ， 子子孙孙 都 受益 。 ” “ 千红万紫 安排 著 ， 只待 新雷 第一声 ” 。 雄安 新区 设立 元年 ， 迎来 丰厚 回报 。 我们 坚信 ， 在 习近平 新 时代 中国 特色 社会主义 思想 指引 下 ， 雄安 新区 的 建设 和 发展 将 不断 提速 ， 国内外 的 人才 、 技术 、 资金 、 项目 将 如 潮水般 滚涌 而 来 ， “ 世界 眼光 、 国际标准 、 中国 特色 、 高点 定位 ” 的 宏图 定 将 变成 现实 ， 作为 京津冀 协同 发展 国家 战略 新 引擎 的 雄安 新区 必将 汇聚 世界 更 多 目光 ！ 责任编辑 ： 杨群"))
    print(inferred_vector)
    for k,v in d2v.model_dm.docvecs.most_similar(3):
        print(k,v)
        #print(d2v.x_train[int(k.split('_')[1])])
        print(d2v.corpus[int(k.split('_')[1])],v)







