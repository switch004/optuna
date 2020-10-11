# -*- coding: utf-8 -*
import os
import numpy as np
import six
from PIL import Image
from itertools import chain
import functools
import glob
import time
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
import chainer
import optuna
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers, iterators, training, datasets
from chainer.datasets import LabeledImageDataset, tuple_dataset
from chainer.training import extensions, triggers 
from chainer.serializers import npz
from chainer.backends import cuda
from chainer import Function
from chainer import Link, ChainList
from chainer import cuda
from shuffle import shuffle
from augmentation import augmentation
from resize import resize
from excel import change_excel
from edit import edit_excel
from scale_augmentation import scale_data_augmentation
from count_image import count_image
from accuracy import accuracy
import xlwt
import shutil



def image2Train(pathsAndLabels, channels=3):

    allData = []
    count_train_label = np.zeros(G)

    #データの追加(画像ファイル，ラベル)
    for pathsAndLabel in pathsAndLabels:
        path = pathsAndLabel[0]
        label = pathsAndLabel[1]
        imagelist = glob.glob(path + '*')
        count_train_label[int(label)] = len(imagelist)
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)#シャッフル

    #チャンネルが1の時は，画像ファイルとラベルデータに追加
    if channels == 1:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathsAndLabel[0])
            imgData = np.asarray([np.float32(img)/255.0])#画像の正規化
            imageData.append(imgData)
            labelData.append(np.int32(pathsAndLabel[1]))

        threshold = np.int32(len(imageData)/8*7)#データの何割を教師データとテストデータにするか
        train = (imageData[0:threshold], labelData[0:threshold])
        test = (imageData[threshold:], labeData[threshold:])

    else:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathAndLabel[0])

            #新たに追加
            img = np.asarray(img, dtype=np.float32)
            img = img[:, :, ::-1]
            img -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            img = img/255.0
            imageData.append(img)
            labelData.append(np.int32(pathAndLabel[1]))

        dataset = {}
        dataset['train_img'] = np.array(imageData[0:N]).transpose(0,3,1,2)
        dataset['train_label'] = np.array(labelData[0:N])


    return (dataset['train_img'], dataset['train_label']), count_train_label


def image2Test(pathsAndLabels, channels=3):

    allData = []
    count_test_label = np.zeros(G)

    #データの追加(画像ファイル，ラベル)
    for pathsAndLabel in pathsAndLabels:
        path = pathsAndLabel[0]
        label = pathsAndLabel[1]
        imagelist = glob.glob(path + '*')
        count_test_label[int(label)] = len(imagelist)
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)#シャッフル

    #チャンネルが1の時は，画像ファイルとラベルデータに追加
    if channels == 1:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathsAndLabel[0])
            imgData = np.asarray([np.float32(img)/255.0])#画像の正規化
            imageData.append(imgData)
            labelData.append(np.int32(pathsAndLabel[1]))

        threshold = np.int32(len(imageData)/8*7)#データの何割を教師データとテストデータにするか
        train = (imageData[0:threshold], labelData[0:threshold])
        test = (imageData[threshold:], labeData[threshold:])

    else:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathAndLabel[0])

            #新たに追加
            img = np.asarray(img, dtype=np.float32)
            img = img[:, :, ::-1]
            img -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            img = img/255.0

            imageData.append(img)
            labelData.append(np.int32(pathAndLabel[1]))

        dataset = {}
        dataset['test_img'] = np.array(imageData[0:N_test]).transpose(0,3,1,2)
        dataset['test_label'] = np.array(labelData[0:N_test])


    return (dataset['test_img'], dataset['test_label']), count_test_label


#class_labels変更する必要
class VGG(chainer.Chain):
    def __init__(self, n_units6, n_units7, class_labels, pretrained_model='VGG_ILSVRC_16_layers.npz'):
        super(VGG, self).__init__()
        with self.init_scope():
            self.base = BaseVGG()
            self.fc6 = L.Linear(None, n_units6, initialW=chainer.initializers.HeNormal())
            self.fc7 = L.Linear(None, n_units7, initialW=chainer.initializers.HeNormal())
            self.fc8 = L.Linear(None, class_labels, initialW=chainer.initializers.HeNormal())
        npz.load_npz(pretrained_model, self.base)

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h,t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
        return loss

    def predict(self, x):
        h = self.base(x)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h

class BaseVGG(Chain):
    def __init__(self):
        super(BaseVGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
            #self.fc6 = L.Linear(None, 4096)
            #self.fc7 = L.Linear(None, 4096)

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        #print(h)

        #h = F.relu(self.fc6(h))
        #h = F.dropout(h)

        #h = F.relu(self.fc7(h))
        #h = F.dropout(h)

        #h = F.dropout(F.relu(self.fc6(h)))
        #h = F.dropout(F.relu(self.fc7(h)))
        return h

def create_optimizer(trial, model):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


def objective(trial, train, test, dir):
    #trialからパラメータを取得
    n_unit6 = trial.suggest_int('n_unit6', 4, 2048)
    n_unit7 = trial.suggest_int('n_unit7', 4, 2048)
    batch_size = trial.suggest_int('batch_size', 2, 128)
    class_labels = 12
    n_epoch = 20
    gpu = 0

    #モデルを定義
    model = VGG(n_units6 = n_unit6, n_units7 = n_unit7, class_labels = class_labels)
    optimizer = create_optimizer(trial, model)

    cuda.get_device(0).use()
    model.to_gpu(0)

    #trainer周りの設定
    train_iter = iterators.SerialIterator(train, batch_size)
    test_iter = iterators.SerialIterator(test, 1, repeat=False, shuffle=False)


    updater = training.StandardUpdater(train_iter, optimizer, device = gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=dir)


    trainer.extend(integrator)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport(
        ['epoch',
        'main/loss',
        'main/accuracy',
        'test/main/loss',
        'test/main/accuracy',
        'elapsed_time',
        'lr',]))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.Evaluator(test_iter, model, device = gpu), name='test')



    #学習の実行
    trainer.run()

    # accuracyを評価指標として用いる
    return 1 - trainer.observation['test/main/accuracy']

if __name__ == '__main__':

    print('dbファイル消した?')

    #以下パラメータの設定
    N_test = 793#評価用画像枚数(381)

    k = 100 #試行回数
    #n_epoch = 20 #更新回数
    #batch_size = 128 #バッチサイズ
    #cnt_trial = 1#試行回数ごとの写真フォルダ分けに使用
    h = 90#入力画像の高さ
    w = 78#入力画像の幅
    gpu = 0 #GPUを用いるなら0,CPUなら-1

    print('試行回数:' +str(k))

    #以下扱うディレクトリのパス設定
    input_dir = 'C:\\Users\\user\\input'#入力教師用画像フォルダ
    input_testing_dir = 'C:\\Users\\user\\input_test'#入力評価用画像フォルダ
    main_dir = 'C:\\Users\\user\\main'#結果出力フォルダ
    result_dir = 'C:\\Users\\user\\result'

    #main_dirがなければ作成
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    #result_dirがなければ作成
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    start_time = time.clock()
    current_dir = os.path.join(main_dir, 'trial')
    if not os.path.exists(current_dir):
            os.mkdir(current_dir)

    current_result_dir = os.path.join(result_dir, 'trial')
    if not os.path.exists(current_result_dir):
            os.mkdir(current_result_dir)

    N_test = shuffle(input_dir, current_dir, input_testing_dir, flag = 1)
    train_dir = os.path.join(current_dir, 'training')
    test_dir = os.path.join(current_dir, 'testing')
    scale_data_augmentation(os.path.join(train_dir, 'image1'), os.path.join(train_dir, 'image2'), (h,w))

    #resizeの実行
    resize(os.path.join(train_dir, 'image2'), os.path.join(train_dir, 'image3'), h, w)
    resize(os.path.join(test_dir, 'image1'), os.path.join(test_dir, 'image2'), h, w)

    #data augmentationの実行
    augmentation(os.path.join(train_dir, 'image3'), os.path.join(train_dir, 'image4'), (h,w))

    N, G = count_image(os.path.join(train_dir, 'image4'))
    print("N:" + str(N))
    print("N_test:" + str(N_test))

    #画像フォルダのパス
    IMG_DIR_train = os.path.join(train_dir, 'image4')
    IMG_DIR_test = os.path.join(test_dir, 'image2')

    #各グループのパス
    dnames_train = glob.glob('{}/*'.format(IMG_DIR_train))
    #print('dnames_train:' + str(dnames_train))
    dnames_test = glob.glob('{}/*'.format(IMG_DIR_test))
    #print('dnames_test:' + str(dnames_test))

    #画像ファイルパス一覧
    fnames_train = [glob.glob('{}/*.jpg'.format(d)) for d in dnames_train]
    fnames_train = list(chain.from_iterable(fnames_train))
    fnames_test = [glob.glob('{}/*.jpg'.format(d)) for d in dnames_test]
    fnames_test = list(chain.from_iterable(fnames_test))

    # それぞれにフォルダ名から一意なIDを付与し、画像を読み込んでデータセット作成
    labels_train = [os.path.basename(os.path.dirname(fn)) for fn in fnames_train]
    dnames_train = [os.path.basename(d) for d in dnames_train]
    labels_train = [dnames_train.index(l) for l in labels_train]
    d_train = LabeledImageDataset(list(zip(fnames_train, labels_train)))

    labels_test = [os.path.basename(os.path.dirname(fn)) for fn in fnames_test]
    dnames_test = [os.path.basename(d) for d in dnames_test]
    labels_test = [dnames_test.index(l) for l in labels_test]
    d_test = LabeledImageDataset(list(zip(fnames_test, labels_test)))

    def transform(data):
        img, label = data
        img = L.model.vision.vgg.prepare(img, size=(h, w))
        img = img / 255. #正規化する．0〜1に落とし込む
        return img, label

    train = chainer.datasets.TransformDataset(d_train, transform)
    test = chainer.datasets.TransformDataset(d_test, transform)


    #目的関数にパラメータを渡す
    obj = functools.partial(objective, train=train, test=test, dir=current_result_dir)

    #Prunerを作成
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)


    #studyを作成
    study = optuna.study.create_study(storage='sqlite:///optimize_yourei.db', study_name='prune_test', load_if_exists=True)
    study.optimize(obj, n_trials=k)

    # Summaryを出力
    print("[Trial summary]")
    df = study.trials_dataframe()
    state = optuna.structs.TrialState
    print("Copmleted:", len(df[df['state'] == state.COMPLETE]))
    print("Pruned:", len(df[df['state'] == state.PRUNED]))
    print("Failed:", len(df[df['state'] == state.FAIL]))

    # 最良のケース
    print("[Best Params]")
    best = study.best_trial
    print("Accuracy:", 1 - best.value)
    print("Batch size:", best.params['batch_size'])
    print("N unit6:", best.params['n_unit6'])
    print("N unit7:", best.params['n_unit7'])

    print()
    print('Params: ')
    for key, value in best.params.items():
        print('{}:{}'.format(key, value))

    print()

    end_time = time.clock()
    print(end_time - start_time)
