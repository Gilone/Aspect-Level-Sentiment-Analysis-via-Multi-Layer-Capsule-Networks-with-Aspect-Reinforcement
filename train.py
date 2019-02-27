import os
import argparse
import numpy as np
import keras.backend as K
from sklearn import metrics
from data_utils import ATSADatesetReader
from models import Caps_LSTM
import pandas as pd


def unpack_data(absa_dataset, one_hot=True):

    from keras.utils import to_categorical
    train_data = absa_dataset.train_data
    test_data = absa_dataset.test_data
    train_indicies = [np.hstack([data['text_raw_indices'], data['text_weight']]) for data in train_data]
    test_indices = [np.hstack([data['text_raw_indices'], data['text_weight']]) for data in test_data]

    if one_hot:
        train_polarity = [to_categorical(data['polarity'], 3) for data in train_data]
        test_polarity = [to_categorical(data['polarity'], 3) for data in test_data]
    else:
        train_polarity = [data['polarity'] for data in train_data]
        test_polarity = [data['polarity'] for data in test_data]
    train_rep = [data['text_rep'] for data in train_data]
    test_rep = [data['text_rep'] for data in test_data]

    return (train_indicies, train_polarity, train_rep), (test_indices, test_polarity, test_rep)


def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 1. - y_pred)) + \
        .5 * (1 - y_true) * K.square(K.maximum(0., y_pred - .1))

    return K.mean(K.sum(L, 1))


def recon_loss(y_true, y_pred):

    L = y_true*K.l2_normalize(y_pred, 1)

    return -K.mean(K.sum(L, 1))


def train(model, absa_dataset, opt):
    from keras import callbacks
    for i in range(5):
        (train_indices, train_polarity, train_rep), \
        (test_indices, test_polarity, test_rep) = unpack_data(absa_dataset)

        log = callbacks.CSVLogger(opt.weights_dir + '/repeats' + str(i)+'/log.csv')
        checkpoint = callbacks.ModelCheckpoint(opt.weights_dir + '/repeats' + str(i) + '/weights-{epoch:02d}.h5',
                                               monitor='val_prob_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)

        model.compile(optimizer=opt.optimizer,
                      loss=[margin_loss, recon_loss, recon_loss],
                      loss_weights=[1., opt.lam, -opt.lam],
                      metrics={'prob': ['accuracy']})
        model.fit([train_indices, train_polarity], [train_polarity, train_rep, train_rep],
                  validation_data=[[test_indices, test_polarity], [test_polarity, test_rep, test_rep]],
                  batch_size=opt.batch_size, epochs=opt.num_epoch, callbacks=[log, checkpoint], verbose=1)
        model.save_weights(opt.weights_dir + '/repeats' + str(i) + '/trained_model.h5')
        print('Trained model saved to \'%s_trained_model.h5\'' % opt.weights_dir)
    return model


def test(eval_model, absa_dataset, opt):
    eval_model.load_weights(opt.weights_dir + opt.weights_file)
    (_, _, _,), \
    (test_indices, test_polarity, _) = unpack_data(absa_dataset, one_hot=False)
    main_outputs, recon1, recon2 = eval_model.predict((np.array(test_indices)))
    predict = np.argmax(main_outputs, axis=1)
    accuracy = np.sum(predict == test_polarity) / len(test_polarity)
    print("accuracy: ", accuracy)
    f1 = metrics.f1_score(test_polarity, predict, labels=[0, 1, 2], average='macro')
    print("f1 score: ", f1)
    result = pd.DataFrame({'max_t_output_polarity_all': predict,
                           'max_t_targets_all': test_polarity,
                           'max_test_acc': accuracy,
                           'max_f1': f1})
    result.to_csv(opt.weights_dir + "test" + ".csv", index=False, sep=',')


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--weights_dir', default='./results/restaurant/caps_lstm_loc1', type=str)
    parser.add_argument('--weights_file', default='/weights-0c.h5', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lam', default=0.003, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--logdir', default='results', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=75, type=int)
    parser.add_argument('--num_class', default=3, type=int)
    parser.add_argument('--num_routing', default=3, type=int)
    opt = parser.parse_args()

    absa_dataset = ATSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
    train_data = absa_dataset.train_data
    test_data = absa_dataset.test_data

    # for i in range(3,6):
        # print(len(train_data), len(test_data))
        # print(train_data[i]["text_raw"])
        # print(train_data[i]["text_raw_indices"])
        # print(train_data[i]["aspect"])
        # print(train_data[i]["aspect_indices"])
        # print(train_data[i]["aspect_in_text"])
        # print(train_data[i]["polarity"])
        # print(train_data[i]["text_weight"])
    if not os.path.exists(opt.weights_dir):
        os.makedirs(opt.weights_dir)
    absa_dataset = ATSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)

    model, eval_model = Caps_LSTM(embedding_matrix=absa_dataset.embedding_matrix,
                                  max_seq_len=opt.max_seq_len, embed_dim=opt.embed_dim,
                                  num_class=opt.num_class, num_routing=opt.num_routing, use_location=True)
    model.summary()

    if opt.mode == 'train':
        train(model, absa_dataset, opt)
    elif opt.mode == 'test':
        test(eval_model, absa_dataset, opt)



