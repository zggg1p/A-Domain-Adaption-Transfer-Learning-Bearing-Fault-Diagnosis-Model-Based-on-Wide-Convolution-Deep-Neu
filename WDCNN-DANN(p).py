# -*- coding: utf-8 -*-
# @Time    : 2022-03-09 21:51
# @Author  : 袁肖瀚
# @FileName: WDCNN-DANN.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn as nn
import argparse
from model import WDCNN1
from torch.nn.init import xavier_uniform_
import torch.utils.data as Data
import matplotlib.pylab as plt
import wandb
import os
from matplotlib.ticker import FuncFormatter

#定义wandb参数
hyperparameter_defaults = dict(
    epochs=70,
    batch_train=40,
    batch_val=50,
    batch_test=40,
    lr=0.0002,
    weight_decay=0.0005,
    r=0.02
)

wandb.init(config=hyperparameter_defaults, project="WDCNN-DANN_test")
config = wandb.config


plt.rcParams['font.family'] = ['Times New Roman']

def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

# model initialization  参数初始化
def weight_init(m):
    class_name = m.__class__.__name__  #得到网络层的名字
    if class_name.find('Conv') != -1:   # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)

def batch_norm_init(m):

    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()


# split train and split data
def data_split_train(data_set, label_set):
    data_set_train = []
    data_set_val = []
    label_set_train = []
    label_set_val = []

    for i in range(data_set.shape[0]):  #行数   shape[2]通道数
        index = np.arange(data_set.shape[1])  #列数矩阵[0 1 2 ''']
        np.random.shuffle(index)  #随机打乱数据 每次shuffle后数据都被打乱，这个方法可以在机器学习训练的时候在每个epoch结束后将数据重新洗牌进入下一个epoch的学习
        a = index[:int((data_set.shape[1]) * 0.8)]
        data = data_set[i]  #第i行

        data_train = data[a]
        data_val = np.delete(data, a, 0)
        data_set_train.append(data_train)
        data_set_val.append(data_val)
        label_set_train.extend(label_set[i][:len(data_train)])
        label_set_val.extend(label_set[i][:len(data_val)])
    data_set_train = np.array(data_set_train).reshape(-1, data_set.shape[-1])
    data_set_val = np.array(data_set_val).reshape(-1, data_set.shape[-1])
    label_set_train = np.array(label_set_train)
    label_set_val = np.array(label_set_val)

    return data_set_train, data_set_val, label_set_train, label_set_val


# training process
def train(train_dataset, val_dataset_s, val_dataset_t,train_dataset_t):
    global alpha
    #torch.cuda.empty_cache()

    length = len(train_dataset.tensors[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=config.batch_train, shuffle=True)

    val_dataloader_s = Data.DataLoader(val_dataset_s, batch_size=config.batch_val, shuffle=False)
    val_dataloader_t = Data.DataLoader(val_dataset_t, batch_size=config.batch_val, shuffle=False)

    t_loader = Data.DataLoader(train_dataset_t, batch_size=int(config.batch_train), shuffle=True)  # 修改这里，保证两个训练集的迭代次数一致
    # t_loader_iter = iter(t_loader)

    val_loss_s = []
    val_loss_t = []
    val_acc_s = []
    val_acc_t = []
    cross_loss = [] #暂时不知道作用
    Source_Train_Acc=[]

    for epoch in range(config.epochs):
        # t_loader = Data.DataLoader(train_dataset_t, batch_size=int(args.batch_train),shuffle=True)  # 修改这里，保证两个训练集的迭代次数一致
        t_loader_iter = iter(t_loader)

        model.train()
        for index, (s_data_train, s_label_train) in enumerate(train_dataloader):
            p = float(index) / 20
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            t_data_train = t_loader_iter.next()
            s_data_train = s_data_train.float().to(device).unsqueeze(dim=1)
            t_data_train = t_data_train[0].float().to(device).unsqueeze(dim=1)
            s_label_train = s_label_train.long().to(device)

            s_domain_label = torch.zeros(config.batch_train).long().cuda()
            t_domain_label = torch.ones(config.batch_train).long().cuda()

            s_out_train, s_domain_out = model(s_data_train, alpha)
            t_out_train, t_domain_out = model(t_data_train, alpha)


            loss_domain_s = criterion(s_domain_out, s_domain_label) #源域域分类损失
            loss_domain_t = criterion(t_domain_out, t_domain_label) #目标域域分类损失

            loss_c = criterion(s_out_train, s_label_train) #分类器损失
            loss = loss_c + (loss_domain_s + loss_domain_t)*0.02


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_s = torch.argmax(s_out_train.data, 1)  # 返回指定维度最大值的序号 dim=1
            correct_s = pred_s.eq(s_label_train).cpu().sum() #源域正确率
            acc = 100. * correct_s.item() / len(s_data_train)
            Source_Train_Acc.append(acc)
            wandb.log({"Source Train Acc": acc})

            if index % 2 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)] \t Loss_c: {:.6f}  Loss_d: {:.6f} Source Train Acc: {:.2f}%'.format
                      (epoch, config.epochs, (index + 1) * len(s_data_train), length,
                       100. * (config.batch_train * (index + 1) / length), loss_c.item(),
                       loss_domain_s.item() + loss_domain_t.item()
                       , acc))

        #validation
        model.eval()
        #源域验证
        correct_val_s = 0
        sum_loss_s = 0
        length_val_s = len(val_dataset_s)
        for index, (s_data_val, s_label_val) in enumerate(val_dataloader_s):
            with torch.no_grad():
                s_data_val = s_data_val.float().to(device).unsqueeze(dim=1)
                s_label_val = s_label_val.long().to(device)

                output_val_s, _ = model(s_data_val, alpha)
                loss_s = criterion(output_val_s, s_label_val)

                pred_val_s = torch.argmax(output_val_s.data, 1)
                correct_val_s += pred_val_s.eq(s_label_val).cpu().sum()
                sum_loss_s += loss_s
        acc_s = 100. * correct_val_s.item() / length_val_s #源域正确率
        average_loss_s = sum_loss_s.item() / length_val_s  #源域损失

        #目标域验证
        correct_val_t = 0
        sum_loss_t = 0
        length_val_t = len(val_dataset_t)
        for index, (t_data_val, t_label_val) in enumerate(val_dataloader_t):
            with torch.no_grad():
                t_data_val = t_data_val.float().to(device).unsqueeze(dim=1)
                t_label_val = t_label_val.long().to(device)

                output_val_t, _ = model(t_data_val, alpha)
                loss_t = criterion(output_val_t, t_label_val)

                pred_val_t = torch.argmax(output_val_t.data, 1)
                correct_val_t += pred_val_t.eq(t_label_val).cpu().sum()
                sum_loss_t += loss_t
        acc_t = 100. * correct_val_t.item() / length_val_t #目标域正确率
        average_loss_t = sum_loss_t.item() / length_val_t  #目标域损失

        metrics = {"Acc_val_t": acc_t, 'epoch':epoch}
        wandb.log(metrics)


        print('\n The {}/{} epoch result : Average loss_s: {:.6f}, Acc_val_s: {:.2f}% , Average loss_t: {:.6f}, Acc_val_t: {:.2f}%'.format(
            epoch, config.epochs, average_loss_s, acc_s,average_loss_t, acc_t))

        val_loss_s.append(loss_s.item())
        val_loss_t.append(loss_t.item())
        val_acc_t.append(acc_t)
        val_acc_s.append(acc_s)

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth"))

    #画出验证集正确率曲线
    plt.plot(val_acc_s, 'r-',marker='s')
    plt.plot(val_acc_t, 'g-',marker='*')
    plt.legend(["Source domain validation accuracy", "Target domain validation accuracy"])
    plt.xlabel('Epochs')
    plt.ylabel('validation accuracy')
    plt.title('Source doamin & Target domain Validation Accuracy Rate')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.savefig("Source doamin & Target domain Validation Accuracy Rate.png")
    plt.show()

    #画出验证集损失
    plt.plot(val_loss_s, 'r-',marker='o')
    plt.plot(val_loss_t, 'g-',marker='x')
    plt.legend(["Source domain validation Loss", "Target domain validation Loss"])
    plt.xlabel('Epochs')
    plt.ylabel('val_loss')
    plt.title('Source domain & Target domain Validation Loss')
    plt.savefig("Source domain & Target domain Validation Loss")
    plt.show()


# testing
def test(test_dataset):
    model.eval()
    length = len(test_dataset)
    correct = 0
    test_loader = Data.DataLoader(test_dataset, batch_size=config.batch_test, shuffle=False)

    y_test = []
    y_pred = []

    for index, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data = data.float().to(device)
            label = label.long().to(device)
            y_test.append(label)

            output, _ = model(data.unsqueeze(dim=1), alpha)
            pred = torch.argmax(output.data, 1)
            y_pred.append(pred)
            correct += pred.eq(label).cpu().sum()

    acc = 100. * correct / length
    return acc


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # use cpu or gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # CWRU
    dataset_s_train = np.load(r'bearing numpy data\dataset_train_0HP_100.npz')
    dataset_s_test = np.load(r'bearing numpy data\dataset_val_0HP_80.npz')
    dataset_t_train = np.load(r'bearing numpy data\dataset_train_3HP_100.npz')
    dataset_t_test = np.load(r'bearing numpy data\dataset_val_3HP_80.npz')

    data_s_train_val = dataset_s_train['data']
    data_s_test = dataset_s_test['data'].reshape(-1, 1024)
    data_t_train_val = dataset_t_train['data']
    data_t_test = dataset_t_test['data'].reshape(-1, 1024)
    label_s_train_val = dataset_s_train['label']
    label_s_test = dataset_s_test['label'].reshape(1, -1)
    label_t_train_val = dataset_t_train['label']
    label_t_test = dataset_t_test['label'].reshape(1, -1)

    iteration_acc = []

    test_acc_s = []


    # repeat several times for an average result
    for iteration in range(1):
        # load model
        model = WDCNN1(C_in=1, class_num=10).to(device)
        model.apply(weight_init)
        model.apply(batch_norm_init)

        # train/val
        data_s_train, data_s_val, label_s_train, label_s_val = data_split_train(data_s_train_val, label_s_train_val)
        data_t_train, data_t_val, _, label_t_val = data_split_train(data_t_train_val, label_t_train_val)

        # transfer ndarray to tensor
        data_s_train = torch.from_numpy(data_s_train)
        data_s_val = torch.from_numpy(data_s_val)
        data_t_val = torch.from_numpy(data_t_val)  #加的验证
        data_s_test = torch.from_numpy(data_s_test)

        data_t_train = torch.from_numpy(data_t_train)
        data_t_test = torch.from_numpy(data_t_test)

        label_s_train = torch.from_numpy(label_s_train)
        label_s_val = torch.from_numpy(label_s_val)
        label_t_val = torch.from_numpy(label_t_val)   #加的验证
        label_s_test = torch.from_numpy(label_s_test)
        #label_t_train = torch.from_numpy(label_t_train)
        label_t_test = torch.from_numpy(label_t_test)

        # seal to data-set
        train_dataset_s = Data.TensorDataset(data_s_train, label_s_train)
        train_dataset_t = Data.TensorDataset(data_t_train)
        val_dataset_s = Data.TensorDataset(data_s_val, label_s_val)
        val_dataset_t = Data.TensorDataset(data_t_val, label_t_val)     #加的验证
        test_dataset_s = Data.TensorDataset(data_s_test, label_s_test.squeeze())
        test_dataset_t = Data.TensorDataset(data_t_test, label_t_test.squeeze())

        # print(train_dataset_s, val_dataset_s)
        criterion = nn.NLLLoss()

        train(train_dataset_s, val_dataset_s, val_dataset_t,train_dataset_t)
        s_test_acc = test(test_dataset_s)
        t_test_acc = test(test_dataset_t)
        print('\n source_acc: {:.2f}% target_acc: {:.2f}%'.format(s_test_acc, t_test_acc))

    wandb.finish()


