import tensorflow as tf
import os
import numpy as np
import pandas as pd
from model import Model
from data_loader import train_random_neighbor, test_random_neighbor


def train(args, data, show_loss):
    train_data, eval_data, test_data = data[0], data[1], data[2]
    train_user_news, train_news_user, test_user_news, test_news_user = data[3], data[4], data[5], data[6]
    news_title, news_entity, news_group = data[7], data[8], data[9]

    checkpt_file = os.path.join(args.save_path, str(args.balance)+'-'+str(args.version)+'-'+'model.ckpt')

    print(len(train_user_news))

    model = Model(args, news_title, news_entity, news_group, len(train_user_news), len(news_title))

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver = tf.train.Saver()
        file = open("local-" + "balance" + str(args.balance) + ".txt", "a")
        global_step = 0
        for step in range(args.n_epochs):
            np.random.shuffle(train_data)
            start = 0
            max_f1 = 0
            curr_step = 0
            patience = 5
            vlss_mn=np.inf
            user_news, news_user = train_random_neighbor(args, train_user_news, train_news_user, len(news_title))
            print(len(user_news))

            # skip the last incomplete minibatch if its size < batch size
            print(train_data.shape[0])

            while start + args.batch_size <= train_data.shape[0]:

                global_step += 1

                _, loss, n, u, train_auc, train_f1 = model.train(sess, get_feed_dict(model, train_data, start,
                                                                                start + args.batch_size,
                                                                                0.5, user_news, news_user))
                start += args.batch_size

                if start % (128*100) == 0:
                    sam_test_user_news, sam_test_news_user = test_random_neighbor(args, test_user_news, test_news_user, len(news_title))
                    print(len(sam_test_news_user))

                    eval_auc, eval_f1, pre = ctr_eval(sess, model, eval_data, args.batch_size, args,
                                                                                  sam_test_user_news,
                                                                                  sam_test_news_user)
                    print("----------\n\n")
                    print('train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f p: %.4f r=: %.4f'
                          % (train_auc, train_f1, eval_auc, eval_f1, p1, r1))

                # if show_loss:
                    print(start, loss)

                    file.write(str(eval_auc)+" " +str(eval_f1) + "\n")
                    file.write(str(start) + " " + str(loss) + "\n")
                    if eval_f1 >= max_f1:
                        saver.save(sess, checkpt_file)
                        max_f1 = eval_f1

                        curr_step = 0

                    else:
                        curr_step += 1
                        if curr_step == patience:
                            print('Early stop valid max f1: ', max_f1)
                            break
            saver.restore(sess, checkpt_file)
            # CTR evaluation
            max_test_user_news, max_test_news_user = test_random_neighbor(args, test_user_news, test_news_user, len(news_title))
            train_auc, train_f1, pre = ctr_eval(sess, model, train_data[:2048], args.batch_size, args, max_user_news, max_news_user)

            eval_auc, eval_f1, pre = ctr_eval(sess, model, eval_data, args.batch_size, args, max_test_user_news, max_test_news_user)
            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1))
            # print('epoch %d    eval auc: %.4f  f1: %.4f'
            #       % (step, eval_auc, eval_f1,))

            test_auc, test_f1, pred = ctr_eval(sess, model, test_data, args.batch_size, args, max_test_user_news, max_test_news_user)
            with open('predict.txt', 'a') as f:
                f.write(str(pred) + "\n")

            print('test auc: %.4f  f1: %.4f'
                  % (test_auc, test_f1))
            file.write("\n-----------\n"+str(step) + "\n" + str(train_auc)+ " " + str(train_f1)+ " " + str(eval_auc)+ " "+str(eval_f1) + " " + str(test_auc)+ " " + str(test_f1) + "\n")
        file.close()

def test(args, data):
    test_data = data[2]
    train_user_news, train_news_user, test_user_news, test_news_user = data[3], data[4], data[5], data[6]

    news_title, news_entity, news_group = data[7], data[8], data[9]
    test_data_old,test_data_new = data[12],data[13]
    model = Model(args, news_title, news_entity, news_group, len(train_user_news), len(news_title))

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        moudke_file = tf.train.latest_checkpoint(args.save_path)
        saver.restore(sess, moudke_file)
        test_auc, test_f1, test_p, test_r = ctr_eval(sess, model, test_data, args.batch_size, args, test_user_news, test_news_user)
        # test_auc_1, test_f1_1,test_p1,test_r1 = ctr_eval(sess, model, test_data_old, args.batch_size, args, test_user_news, test_news_user,
        #                              topic_news, len(news_title))
        # test_auc_2, test_f1_2,test_p2,test_r2 = ctr_eval(sess, model, test_data_new, args.batch_size, args, test_user_news, test_news_user,
        #                              topic_news, len(news_title))
        print('test auc: %.4f  f1: %.4f  p: %.4f  r: %.4f' % (test_auc, test_f1,test_p,test_r))
        # print('old test auc: %.4f  f1: %.4f  p: %.4f  r: %.4f' % (test_auc_1, test_f1_1,test_p1,test_r1))
        # print('new test auc: %.4f  f1: %.4f  p: %.4f  r: %.4f' % (test_auc_2, test_f1_2,test_p2,test_r2))


def get_feed_dict(model, data, start, end, dropout, user_news, news_user):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.news_indices: data[start:end, 1],
                 model.dropout_rate: dropout,
                 model.labels: data[start:end, 3],
                 model.user_news: user_news,
                 model.news_user: news_user}
    return feed_dict


def ctr_eval(sess, model, data, batch_size, args, input_user_news, input_news_user):
    start = 0
    auc_list = []
    user_f1 = []
    f1_list = []
    pre_list = []
    news_rep = []
    user_rep = []
    los_list = []

    user_news, news_user = input_user_news, input_news_user
    while start + batch_size <= data.shape[0]:

        auc, f1, predict = model.eval(sess, get_feed_dict(model, data, start, start + batch_size, 0, user_news, news_user))
        auc_list.append(auc)
        f1_list.append(f1)
        pre_list.append(predict)

        start += batch_size
        # print("!!!!")
        # print(scores[:30],l[:30])
        # with open('eval_re1.txt','a') as f:
        #     f.write(str(scores)+"\n"+str(l)+'----')

    return float(np.mean(auc_list)), float(np.mean(f1_list)), pre_list[0]