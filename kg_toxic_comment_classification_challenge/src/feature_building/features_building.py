#!/usr/bin/python
import re
from functools import partial

import datetime
from keras.preprocessing.text import maketrans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, auc, roc_curve

from commons import DATA_DIR, DATA_ORI_DIR, DATA_FEATURES_DIR
import pandas as pd
import os

TEST_CSV_PATH = DATA_ORI_DIR + 'test.csv'
TRAIN_CSV_PATH = DATA_ORI_DIR + 'train.csv'
SAMPLE_SUBMISSION_CSV_PATH = DATA_ORI_DIR + 'sample_submission.csv'


def first_check():
    '''
    train_ori_df中共有159571条记录, 8列分别为
    'id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate'
    其中 id 列为唯一标示
    :return:
    '''
    train_ori_df = pd.read_csv(TRAIN_CSV_PATH)
    train_ori_df.columns
    # Index(['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate'],
    #       dtype='object')
    train_ori_df.shape
    # (159571, 8)
    train_ori_df['id'].unique
    # Name: id, Length: 159571, dtype: object
    train_label_cols = list(train_ori_df.columns)[2:]
    for col in train_label_cols:
        print(col)
        print(train_ori_df[col].value_counts(dropna=False))

    # Name: toxic, dtype: int64
    # 0   144277
    # 1   15294
    #
    # Name: severe_toxic, dtype: int64
    # 0    157976
    # 1    1595
    #
    # Name: obscene, dtype: int64
    # 0    151122
    # 1    8449
    #
    # Name: threat, dtype: int64
    # 0    159093
    # 1    478
    #
    # Name: insult, dtype: int64
    # 0    151694
    # 1    7877
    #
    # Name: identity_hate, dtype: int64
    # 0    158166
    # 1    1405


    test_ori_df = pd.read_csv(TEST_CSV_PATH)
    test_ori_df.columns
    # Index(['id', 'comment_text'], dtype='object')
    test_ori_df.shape
    # (153164, 2)

    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
    sample_submission_df.columns
    # Index(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate'],dtype='object')
    sample_submission_df.shape
    # (153164, 7)

    type(train_ori_df)


def comment_2_vec_factory(category):
    """
    convert comment into vector
    :param category:
    :return:
    """
    assert category in ['tf-idf']
    pass


def clean_comment(row):
    """
    clean comment.
    step 1 : lower all words
    step 2 : remove all marks
    :param :
    :return:
    """
    comment = row['comment_text']
    comment = comment.lower()  # convert <comment_text> into lower case;
    chars_list = [c for c in list(comment) if re.match('[a-z\s]', c)]  # remove all char which is not character or space
    comment = ''.join(chars_list)  # re-union with space
    trans_chars = "'.!_,$%^*()+\"\\\n+——！，。？?、~@#￥%……&*（）"
    trans_tbl = str.maketrans(trans_chars, ''.join([' ' for i in range(len(trans_chars))]))
    comment = comment.translate(trans_tbl)
    row['comment_text'] = comment.strip()
    return row


def epoch_train(train_df):
    type_cols = train_df.columns.tolist()
    type_cols.remove('comment_text')

    for type in type_cols:
        pass


def main():
    train_ori_df = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
    test_ori_df = pd.read_csv(TEST_CSV_PATH, index_col='id')

    train_part_comment_vectorized_df_path = DATA_FEATURES_DIR + 'train_part_comment_vectorized.csv'
    test_part_comment_vectorized_df_path = DATA_FEATURES_DIR + 'test_part_comment_vectorized.csv'
    test_comment_vectorized_df_path = DATA_FEATURES_DIR + 'test_comment_vectorized.csv'
    train_part_comment_cleaned_path = DATA_FEATURES_DIR + 'train_part_comment_cleaned.csv'
    test_part_comment_cleaned_path = DATA_FEATURES_DIR + 'test_part_comment_cleaned.csv'
    test_comment_cleaned_path = DATA_FEATURES_DIR + 'test_comment_cleaned.csv'

    ready_flag = True
    if not os.path.exists(train_part_comment_vectorized_df_path):
        ready_flag = False
    if not os.path.exists(test_part_comment_vectorized_df_path):
        ready_flag = False
    if not os.path.exists(test_comment_vectorized_df_path):
        ready_flag = False

    ready_flag = False
    if not ready_flag:
        train_size = 0.7
        train_part, test_part = train_test_split(train_ori_df, train_size=train_size)
        # clean the original comment text
        clean_comment_p = partial(clean_comment)
        train_part_comment_cleaned = train_part.apply(clean_comment_p, axis=1)
        train_part_comment_cleaned.to_csv(train_part_comment_cleaned_path)
        test_part_comment_cleaned = test_part.apply(clean_comment_p, axis=1)
        test_part_comment_cleaned.to_csv(test_part_comment_cleaned_path)
        test_comment_cleaned = test_ori_df.apply(clean_comment_p, axis=1)
        test_comment_cleaned.to_csv((test_comment_cleaned_path))

        # convert comment into vector, max_features set 1000
        max_features = 1000
        count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english',
                                    max_features=max_features)

        comment_corpus = pd.Series(
            train_part_comment_cleaned['comment_text'].tolist() +
            test_part_comment_cleaned['comment_text'].tolist() +
            test_comment_cleaned['comment_text'].tolist()).astype(str)
        count_vec.fit(comment_corpus)

        train_part_comment_vectorized = count_vec.transform(train_part_comment_cleaned['comment_text'])
        train_part_comment_vectorized_df = pd.DataFrame(train_part_comment_vectorized.toarray(),
                                                        columns=count_vec.get_feature_names(),
                                                        index=train_part_comment_cleaned.index)
        train_part_comment_vectorized_df.to_csv(path_or_buf=train_part_comment_vectorized_df_path)

        test_part_comment_vectorized = count_vec.transform(test_part_comment_cleaned['comment_text'])
        test_part_comment_vectorized_df = pd.DataFrame(test_part_comment_vectorized.toarray(),
                                                       columns=count_vec.get_feature_names(),
                                                       index=test_part_comment_cleaned.index)
        test_part_comment_vectorized_df.to_csv(path_or_buf=test_part_comment_vectorized_df_path)

        test_comment_vectorized = count_vec.transform(test_comment_cleaned['comment_text'])
        test_comment_vectorized_df = pd.DataFrame(test_comment_vectorized.toarray(),
                                                  columns=count_vec.get_feature_names(),
                                                  index=test_comment_cleaned.index)
        test_comment_vectorized_df.to_csv(path_or_buf=test_comment_vectorized_df_path)
    else:
        train_part_comment_vectorized_df = pd.read_csv(train_part_comment_vectorized_df_path, index_col='id')
        test_part_comment_vectorized_df = pd.read_csv(test_part_comment_vectorized_df_path, index_col='id')
        test_comment_vectorized_df = pd.read_csv(test_comment_vectorized_df_path, index_col='id')
        train_part_comment_cleaned = pd.read_csv(train_part_comment_cleaned_path, index_col='id')
        test_part_comment_cleaned = pd.read_csv(test_part_comment_cleaned_path, index_col='id')
        test_comment_cleaned = pd.read_csv(test_comment_cleaned_path, index_col='id')

    types_list = list(train_ori_df.columns)[1:]
    # ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for type_single in types_list:

        ori_df = pd.merge(left=train_part_comment_vectorized_df,
                          right=train_part_comment_cleaned[type_single].to_frame(), left_index=True, right_index=True)

        print('Processing type : %s' % type_single)

        positive_df = ori_df[ori_df[type_single] == 1]
        negative_df = ori_df[ori_df[type_single] == 0]
        time_no = negative_df.shape[0] // positive_df.shape[0]

        final_df = ori_df
        for i in range(time_no):
            final_df = final_df.append(other=positive_df)

        X = final_df.drop(type_single, axis=1)
        y = final_df[type_single]

        # mdl = RandomForestClassifier()
        # mdl = LogisticRegression()
        mdl = SGDClassifier()

        mdl.fit(X=X, y=y)

        prediction_train = mdl.predict(X=X)
        accuracy_train = accuracy_score(y_true=y, y_pred=prediction_train)
        recall_train = recall_score(y_true=y, y_pred=prediction_train)
        print('On train_part -> %s accuracy : %f recall : %f' % (type_single, accuracy_train, recall_train))

        prediction_test = mdl.predict(X=test_part_comment_vectorized_df)
        accuracy_test = accuracy_score(y_true=test_part_comment_cleaned[type_single], y_pred=prediction_test)
        recall_test = recall_score(y_true=test_part_comment_cleaned[type_single], y_pred=prediction_test)
        fpr, tpr, thresholds = roc_curve(y_true=test_part_comment_cleaned[type_single], y_score=prediction_test)
        auc_ = auc(fpr, tpr)

        print('On test_part -> %s accuracy : %f recall : %f auc : %f' % (type_single, accuracy_test, recall_test, auc_))

        prediction = mdl.predict(X=test_comment_vectorized_df)

        prediction_df = pd.DataFrame(data=prediction, index=test_comment_vectorized_df.index)
        prediction_df.columns = [type_single]
        test_ori_df[type_single] = prediction_df

    # test_ori_df = pd.read_csv(DATA_DIR + 'submission_20180304194726.csv',index_col='id')
    # test_ori_df.index= test_ori_df['id']
    test_ori_df = test_ori_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    test_ori_df.to_csv(DATA_DIR + 'submission_%s.csv' % (datetime.datetime.now().strftime('%Y%m%d%H%M%S')))

    # for type_single in types_list:
    #     print('Processing type : %s' % type_single)
    #     print(train_ori_df[type_single].value_counts())


if __name__ == '__main__':
    main()
#
# from sklearn.metrics import log_loss
#
# import numpy as np
#
# max([0, 1, 1, 0, 0])
#
# from nltk.corpus import stopwords
#
# stops = set(stopwords.words('english'))
#
# corpus = ['I love you ', 'You will win this world', 'He loves me ']
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# vectorizer = CountVectorizer()
#
# csr_mat = vectorizer.fit_transform(raw_documents=corpus)
#
# print(type(csr_mat))
#
# print(csr_mat)
#
# print(csr_mat.todense())
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# transformer = TfidfTransformer()
#
# tfidf = transformer.fit_transform(csr_mat)
#
# print(tfidf)
#
# print(tfidf.todense())
#
# tfidf = transformer.fit_transform(corpus)
#
# print(tfidf)
#
# print(tfidf.todense())
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tfidf_vectorizer = TfidfVectorizer()
#
# tfidf_vec = tfidf_vectorizer.fit_transform(corpus)
#
# print(tfidf_vec.todense())
#
# print(type(tfidf_vec.todense()))
