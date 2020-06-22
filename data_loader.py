import numpy as np
import json
import random
import time
import datetime
from scipy import sparse
from collections import defaultdict

USER_NEWS_FILE = '../data/user_news.json'
NEWS_ENTITY_FILE = '../data/news_entity.json'
ENTITY_ENTITY_FILE = '../data/entity_entity.json'

def load_data(args):
    # news, news_title, news_entity, entity_news = load_news(args)
    # train_data, eval_data, test_data, user_news, news_user = load_events(news, args)
    # entity_entity = load_entity()
    r = np.load("../result7.npz", allow_pickle=True)
    train_data = r['train_data']
    test_data = r['test_data']
    user_news = r['user_news']
    news_user = r['news_user']
    news_title = r['news_title']
    news_entity = r['news_entity']
    entity_news = r['entity_news']
    # entity_entity = r['entity_entity']
    entity_entity = []
    clicks = r['clicks']
    np.random.shuffle(test_data)
    np.random.shuffle(test_data)
    # test_data = np.array(test_data)
    np.random.shuffle(train_data)
    eval_indices = np.random.choice(list(range(test_data.shape[0])), size=int(test_data.shape[0] * 0.2), replace=False)
    test_indices = list(set(range(test_data.shape[0])) - set(eval_indices))
    eval_data = test_data[eval_indices]
    test_data = test_data[test_indices]
    # print(eval_indices[:100],test_indices[:100])
    return train_data, eval_data, test_data, user_news, news_user, news_title, news_entity, entity_news, clicks ,entity_entity

def load_new_data(args):
    r = np.load("./data/data.npz", allow_pickle=True)
    train_data = r['train_data']
    test_data = r['test_data']
    news_entity = r['news_entity']
    news_group = r['news_group']
    news_title = r['news_title'][:, :args.title_len]

    with open("./data/train_user_news.txt", 'r') as file:
        train_user_news = eval(file.read())
    print('train_user_news load over!')
    with open("./data/test_user_news.txt", 'r') as file:
        test_user_news = eval(file.read())
    print('test_user_news load over!')
    with open("./data/train_news_user.json", 'r') as file:
        train_news_user = json.load(file)
        train_news_user = dict(zip(list(map(int,train_news_user.keys())),train_news_user.values()))
    print('train_news_user load over!')
    with open("./data/test_news_user.json", 'r') as file:
        test_news_user = json.load(file)
        test_news_user = dict(zip(list(map(int,test_news_user.keys())),test_news_user.values()))
    print('test_news_user load over!')
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    l = int(len(test_data) * 0.1)
    # eval_indices = np.random.choice(list(range(test_data.shape[0])), size=int(test_data.shape[0] * 0.2), replace=False)
    # test_indices = list(set(range(test_data.shape[0])) - set(eval_indices))
    eval_data = test_data[:l]
    test_data = test_data[l:]

    return train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, \
        news_title, news_entity, news_group


def train_random_neighbor(args, train_user_news, train_news_user, news_len):
    user_news = np.zeros([len(train_user_news),args.news_neighbor], dtype=np.int32)
    for i in range(1, len(train_user_news)):
        n_neighbors = len(train_user_news[i])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=False)  #不放回
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=True)
        user_news[i] = np.array([train_user_news[i][k] for k in sampled_indices])

    news_user = np.zeros([news_len, args.user_neighbor], dtype=np.int32)
    for i in train_news_user:
        n_neighbors = len(train_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=True)
        news_user[int(i)] = np.array([train_news_user[i][k] for k in sampled_indices])

    return user_news, news_user


def test_random_neighbor(args, test_user_news, test_news_user, news_len):
    user_news = np.zeros([len(test_user_news),args.news_neighbor], dtype=np.int32)
    for i in range(1, len(test_user_news)):
        n_neighbors = len(test_user_news[i])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=True)
        user_news[i] = np.array([test_user_news[i][k] for k in sampled_indices])

    news_user = np.zeros([news_len, args.user_neighbor], dtype=np.int32)
    for i in test_news_user:
        n_neighbors = len(test_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=True)
        news_user[int(i)] = np.array([test_news_user[i][k] for k in sampled_indices])

    return user_news, news_user

def load_events(news, args):
    with open(USER_NEWS_FILE, 'r') as file:
        users = json.load(file)
    len_news = len(news)
    t_user_news = defaultdict(list)
    t_news_user = defaultdict(list)
    user_news = np.zeros([1+len(users), args.news_neighbor], dtype=np.int32)
    news_user = np.zeros([1+len_news, args.user_neighbor], dtype=np.int32)
    data = []
    for user in users:
        for i in range(len(users[user]) - 1):
            t_user_news[int(user)].append(users[user][i]['id'])
            t_news_user[users[user][i]['id']].append(int(user))

        # sample news neighbors of user
        n_neighbors = len(t_user_news[int(user)])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                               replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                               replace=True)
        user_news[int(user)] = np.array([t_user_news[int(user)][i] for i in sampled_indices])



        t1 = trans_time(users[user][-1]['time'], users[user][-1]['publishtime'])
        data.append([user, users[user][-1]['id'], t1, 1])

        read_news = [x['id'] for x in users[user]]
        negtive = str(random.sample(set(range(1, len_news + 1)) - set(read_news), 1)[0])
        t2 = trans_time(news[negtive]['time'], news[negtive]['publishtime'])
        data.append([user, negtive, t2, 0])
    # sample user neighbors of news
    for i in range(1,len(t_news_user)+1):
        n_neighbors = len(t_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                               replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                               replace=True)
        news_user[i] = np.array([t_news_user[i][j] for j in sampled_indices])

    # dataset split
    train_data, eval_data, test_data = dataset_split(np.array(data), args)

    return train_data, eval_data, test_data, user_news, news_user


def load_news(args):
    with open(NEWS_ENTITY_FILE,'r') as file:
        news = json.load(file)
    news_title = []
    n_entity = 69473
    t_entity_news = defaultdict(list)
    entity_news = np.zeros([1 + n_entity, args.news_neighbor], dtype=np.int64)
    news_entity = np.zeros([1 + len(news), args.entity_neighbor], dtype=np.int64)
    for i in range(1, len(news) + 1):

        if len(news[str(i)]['title']) <= args.title_len:
            news_title.append(news[str(i)]['title'].extend([0]*(args.title_len-len(news[str(i)]['title']))))
        else:
            news_title.append(news[str(i)]['title'][:args.title_len])
        # sample entity neighbors of news
        n_neighbors = len(news[str(i)]['entity'])
        if n_neighbors >= args.entity_neighbor:

            sampled_indices = np.random.choice(list(range(args.entity_neighbor)), size=args.entity_neighbor,
                                               replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.entity_neighbor,
                                               replace=True)
        news_entity[i] = np.array([news[str(i)]['entity'][j] for j in sampled_indices])

        for e in news[str(i)]['entity']:
            t_entity_news[e].append(i)

    # sample news neighbors of entity
    for j in range(1, len(t_entity_news) + 1):
        n_neighbors = len(t_entity_news[j])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                               replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                               replace=True)
        entity_news[j] = np.array([t_entity_news[j][k] for k in sampled_indices])
    news_title = np.array(news_title)

    return news, news_title, news_entity, entity_news


def load_entity():
    with open(ENTITY_ENTITY_FILE,'r') as file:
        entity_entity = json.load(file)

    return entity_entity


def trans_time(linux_time, utc_time):
    UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
    utcTime = datetime.datetime.strptime(utc_time, UTC_FORMAT)
    ans_time = time.mktime(utcTime.timetuple())

    return linux_time - ans_time


def dataset_split(data, args):
    print('splitting dataset ...')

    eval_ratio = (1 - args.ratio)/2
    test_ratio = (1 - args.ratio)/2
    n_samples = data.shape[0]

    eval_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * eval_ratio), replace=False)
    left = set(range(n_samples)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    print(n_samples,len(eval_indices),len(test_indices),len(train_indices))
    train_data = data[train_indices]
    eval_data = data[eval_indices]
    test_data = data[test_indices]

    return train_data, eval_data, test_data
