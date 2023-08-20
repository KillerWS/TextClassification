# TF-IDF是一种特征提取的方法
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

#读取数据
train = pd.read_csv('files/train.csv')
#填充缺失数据为NaN,这里是处理了title和abstract两个数据列
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

# 读取测试集用于测试
test = pd.read_csv('files/testB.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')

# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')

# 引入一个外部专门的停用词列表
stopwords =[i.strip() for i in open(r'files/stop.txt',encoding='utf-8').readlines()]

#使用停用词
vector = TfidfVectorizer(stop_words = stopwords)

#这一步实质上是计算每个词汇的IDF
vector = TfidfVectorizer().fit(train["text"])

print(vector)

# 拟合之后，调用 transform 方法即可得到提取后的特征数据
#train_vector = vector.transform()
#这里是的参数是raw_documents, 这一行将训练数据（train）中的文本转换为TF-IDF特征向量。这个特征向量将用于训练分类模型。
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])

# 引入模型
model = LogisticRegression()

# 开始训练！，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
# 此处的 train_vector 是已经经过特征提取的训练数据!
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)
test['Keywords'] = test['title'].fillna('')
test[['uuid','Keywords','label']].to_csv('TF_IDF2.csv', index=None)
