#coding=utf8
"""
@Author:张前进
@description:使用改进的tf_idf选出特征词构建词典，使用one-hot编码技术(或者使用词频)构建特征向量,使用特征向量训练模型得到相应的模型参数
@Update date：2018.01.02
"""


from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pylab as pl
import math

import jieba

"""
获取停顿词
"""
def get_stopword(filename):
    stopword_list = []
    with open(filename,'rb') as f:
        for line in f:
            line = line.strip()
            stopword_list.append(line.decode('utf-8'))
    return stopword_list


"""
获取每行文本的词集合以及所有文本词的集合，同时剔除停顿词
result_list获得每行文本词的集合的列表
word_set获得所有文本词的集合
"""
def readdata(filename, stopword):
    content_list = []
    word_dict = {}
    result_list = []
    temp_line_list = []
    with open(filename,'rb') as f:
        for line in f:
            temp_set = set()
            line = line.strip()
            if line in temp_line_list:
                continue
            temp_line_list.append(line)
            content_list.append(line.decode("utf-8"))
            segment = jieba.cut(line.decode("utf-8"))
            for ele in segment:
                if ele not in stopword:
                    if ele == " ":
                        continue
                    temp_set.add(ele)
                    word_dict.setdefault(ele,0)
                    word_dict[ele] += 1
            result_list.append(list(temp_set))
    return   result_list, word_dict, content_list

def get_feature_word(bad_word_dict, good_word_dict):
    all_num_bad_word = sum(bad_word_dict[key] for key in bad_word_dict)
    all_num_good_word = sum(good_word_dict[key] for key in good_word_dict)
    tf_bad_word = {}
    idf_bad_word = {}
    tf_good_word = {}
    idf_good_word = {}
    tf_idf_bad_word = {}
    tf_idf_good_word = {}
    for key in bad_word_dict:
        tf_bad_word[key] = bad_word_dict[key]/all_num_bad_word
    for key in good_word_dict:
        tf_good_word[key] = good_word_dict[key]/all_num_good_word
    for key in bad_word_dict:
        idf_bad_word[key] = math.log2(1+(tf_bad_word[key]+0.000001)/(tf_good_word.get(key,0)+0.000001))
        tf_idf_bad_word[key] = tf_bad_word[key]*idf_bad_word[key]
    for key in good_word_dict:
        idf_good_word[key] = math.log2(1 + (tf_good_word[key] + 0.000001) / (tf_bad_word.get(key, 0) + 0.000001))
        tf_idf_good_word[key] = tf_good_word[key] * idf_good_word[key]
    tf_idf_bad_word_sort = sorted(tf_idf_bad_word.items(),key=lambda item:item[1],reverse=True)
    tf_idf_good_word_sort = sorted(tf_idf_good_word.items(),key=lambda item:item[1],reverse=True)
    return tf_idf_bad_word_sort, tf_idf_good_word_sort
"""
获取每个文本词对应的词向量，segment_list为文本词集合列表，all_word_list为词典
"""
def get_word_vec(segment_list, all_word_list):
    word_vec_list = []
    for word_list in segment_list:
        temp_list = []
        for word in all_word_list:
            if word in word_list:
                temp_list.append(1)
            else:
                temp_list.append(0)
        word_vec_list.append(temp_list)
    return word_vec_list


if __name__ == "__main__":
    """
    数据集在路径D:/pythonprogram/nlp/dataset/下
    """
    bad_info_filename = 'D:/pythonprogram/nlp/dataset/bad.txt'
    good_info_filename = "D:/pythonprogram/nlp/dataset/good.txt"
    stop_info_filename = "D:/pythonprogram/nlp/dataset/stop.txt"
    stopword_list = get_stopword(stop_info_filename)
    bad_segment_list, bad_word_dict, bad_content_list = readdata(bad_info_filename, stopword_list)
    good_segment_list, good_word_dict, good_content_list = readdata(good_info_filename, stopword_list)
    """
    获得词典,good_key_word_num正评论关键词的个数，bad_key_word_num负评论关键词的个数
    """
    print(bad_content_list)
    tf_idf_bad_word_sort, tf_idf_good_word_sort = get_feature_word(bad_word_dict, good_word_dict)
    good_key_word_num = 150
    bad_key_word_num = 150
    Word_Dict = []
    word_bag_bad = [item[0] for item in tf_idf_bad_word_sort[:bad_key_word_num]]
    word_bag_good = [item[0] for item in tf_idf_good_word_sort[:good_key_word_num]]
    Word_Dict.extend(word_bag_bad)
    Word_Dict.extend(word_bag_good)

    bad_word_vec = get_word_vec(bad_segment_list, Word_Dict)
    good_word_vec = get_word_vec(good_segment_list, Word_Dict)
    X = []
    Y = []
    content = []
    X.extend(bad_word_vec)
    X.extend(good_word_vec)
    Y.extend([0 for i in range(len(bad_word_vec))])
    Y.extend([1 for i in range(len(good_word_vec))])
    content.extend(bad_content_list)
    content.extend(good_content_list)
    """
    融合X，Y以及X对应的内容，打乱数据，分成训练数据x,y和测试数据x_test,y_test
    """
    X_Y_content = list(zip(X, Y, content))
    X_Y_content_process = shuffle(X_Y_content, random_state=1)
    x_train = []
    y_train = []
    content_train = []
    x_test = []
    y_test = []
    content_test = []
    for X_Y_content_ele in X_Y_content_process:
        x_train.append(X_Y_content_ele[0])
        y_train.append(X_Y_content_ele[1])
        content_train.append(X_Y_content_ele[2])
    for X_Y_content_ele in X_Y_content_process:
        x_test.append(X_Y_content_ele[0])
        y_test.append(X_Y_content_ele[1])
        content_test.append(X_Y_content_ele[2])
    """
    引入模型
    """
    clf = LogisticRegression()
    # x = [[1,2,4],[3,1,2],[2,3,5]]
    # y = [1,0,1]
    """
    模型训练
    """
    print(x_train)
    print(y_train)
    clf.fit(x_train, y_train)
    """
    模型参数
    """
    print("w is")
    print(clf.coef_)
    print("...........")
    print("b is")
    print(clf.intercept_)
    print("...........")
    """
    模型预测
    """
    pred_proba = clf.predict_proba(x_test)
    pred = [ele[1] for ele in pred_proba]
    print("test pred result")
    for i in range(len(x_test)):
        print("%f\t%f\t%s\t%s" % (y_test[i], pred[i], content_test[i], x_test[i]))
    """
    通过metrics.roc_curve获得，假正率fpr和真正率tpr以及相应的阈值thresholds
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    precision, recall, th = metrics.precision_recall_curve(y_test, pred)
    """
    画roc曲线图
    """
    print("auc is %f" % metrics.auc(fpr, tpr))
    plt.plot(fpr, tpr, 'r')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(u'roc curve')
    plt.plot([0, 1], [0, 1], '--')
    plt.grid(True)
    plt.show()


