import torch
import numpy as np
import torch.nn as nn
import argparse
from model_2 import WDCNN2
from torch.nn.init import xavier_uniform_
import torch.utils.data as Data
import matplotlib.pylab as plt
import wandb
import os

hyperparameter_defaults = dict(
    epochs=60,
    batch_train=40,
    batch_val=50,
    batch_test=40,
    lr=0.0002,
    weight_decay=0.0005,
    r=0.02
)

wandb.init(config=hyperparameter_defaults, project="pytorch_WDCNN-DANN")
config = wandb.config


# model initialization  参数初始化
def weight_init(m):
    class_name = m.__class__.__name__  # 得到网络层的名字
    if class_name.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
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

    for i in range(data_set.shape[0]):  # 行数   shape[2]通道数
        index = np.arange(data_set.shape[1])  # 列数矩阵[0 1 2 ''']
        np.random.shuffle(index)  # 随机打乱数据 每次shuffle后数据都被打乱，这个方法可以在机器学习训练的时候在每个epoch结束后将数据重新洗牌进入下一个epoch的学习
        a = index[:int((data_set.shape[1]) * 0.8)]
        data = data_set[i]  # 第i行

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
def train(train_dataset, val_dataset_s, val_dataset_t, train_dataset_t,test_dataset_s,test_dataset_t):
    global alpha
    # torch.cuda.empty_cache()

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
    cross_loss = []
    acc1 = []

    for epoch in range(config.epochs):
        # t_loader = Data.DataLoader(train_dataset_t, batch_size=int(args.batch_train),shuffle=True)  # 修改这里，保证两个训练集的迭代次数一致
        t_loader_iter = iter(t_loader)
        model.train()
        i = 0
        for index, (s_data_train, s_label_train) in enumerate(train_dataloader):
            i += 1
            p = float(i + epoch * config.batch_train) / config.epochs / config.batch_train
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            t_data_train = t_loader_iter.next()
            s_data_train = s_data_train.float().to(device).unsqueeze(dim=1)
            t_data_train = t_data_train[0].float().to(device).unsqueeze(dim=1)
            s_label_train = s_label_train.long().to(device)

            s_domain_label = torch.zeros(config.batch_train).long().cuda()
            t_domain_label = torch.ones(config.batch_train).long().cuda()

            s_out_train, s_domain_out = model(s_data_train, alpha)
            t_out_train, t_domain_out = model(t_data_train, alpha)

            loss_domain_s = criterion(s_domain_out, s_domain_label)
            loss_domain_t = criterion(t_domain_out, t_domain_label)

            loss_c = criterion(s_out_train, s_label_train)
            loss = loss_c + (loss_domain_s + loss_domain_t) * 0.02

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(s_out_train.data, 1)  # 返回指定维度最大值的序号 dim=1
            correct = pred.eq(s_label_train).cpu().sum()
            acc = 100. * correct.item() / len(s_data_train)
            acc1.append(acc)
            wandb.log({"Train_acc": acc})

            if index % 2 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)] \t Loss_c: {:.6f}  Loss_d: {:.6f} Acc: {:.2f}%'.format
                      (epoch, config.epochs, (index + 1) * len(s_data_train), length,
                       100. * (config.batch_train * index / length), loss_c.item(),
                       loss_domain_s.item() + loss_domain_t.item()
                       , acc))
            if i % 2 == 0:
                s_test_acc = test(test_dataset_s)
                t_test_acc = test(test_dataset_t)
                metrics = {"t_test_acc": t_test_acc}
                wandb.log(metrics)
                # print('source_acc: {:.2f}% target_acc: {:.2f}%'.format(s_test_acc, t_test_acc))
                model.train()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


# testing
def test(test_dataset):
    model.eval()
    length = len(test_dataset)
    correct = 0
    test_loader = Data.DataLoader(test_dataset, batch_size=config.batch_test, shuffle=False)
    for index, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data = data.float().to(device)
            label = label.long().to(device)

            output, _ = model(data.unsqueeze(dim=1), alpha)
            pred = torch.argmax(output.data, 1)
            correct += pred.eq(label).cpu().sum()

    Test_acc = 100. * correct / length

    return Test_acc


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # use cpu or gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)
    # load data
    # dataset_s_train = np.load(r'E:\智能制造竞赛\fault diagnosis  Code_桌面\LinGang2\0.2-1200-dataset_train.npz')
    # dataset_s_test = np.load(r'E:\智能制造竞赛\fault diagnosis  Code_桌面\LinGang2\0.2-1200-dataset_val.npz')
    # dataset_t_train = np.load(r'E:\智能制造竞赛\fault diagnosis  Code_桌面\LinGang2\0.2-1800-dataset_train.npz')
    # dataset_t_test = np.load(r'E:\智能制造竞赛\fault diagnosis  Code_桌面\LinGang2\0.2-1800-dataset_val.npz')

    # CWRU
    dataset_s_train = np.load(r'G:\bearing numpy data\dataset_train_1HP_100.npz')
    dataset_s_test = np.load(r'G:\bearing numpy data\dataset_val_1HP_80.npz')
    dataset_t_train = np.load(r'G:\bearing numpy data\dataset_train_2HP_100.npz')
    dataset_t_test = np.load(r'G:\bearing numpy data\dataset_val_2HP_80.npz')

    data_s_train_val = dataset_s_train['data']
    data_s_test = dataset_s_test['data'].reshape(-1, 1024)
    data_t_train_val = dataset_t_train['data']
    data_t_test = dataset_t_test['data'].reshape(-1, 1024)
    label_s_train_val = dataset_s_train['label']
    label_s_test = dataset_s_test['label'].reshape(1, -1)
    label_t_train_val = dataset_t_train['label']
    label_t_test = dataset_t_test['label'].reshape(1, -1)

    # repeat several times for an average result
    for iteration in range(1):
        # load model
        model = WDCNN2(C_in=1, class_num=10).to(device)
        model.apply(weight_init)
        model.apply(batch_norm_init)

        # train/val
        data_s_train, data_s_val, label_s_train, label_s_val = data_split_train(data_s_train_val, label_s_train_val)
        data_t_train, data_t_val, _, label_t_val = data_split_train(data_t_train_val, label_t_train_val)

        # transfer ndarray to tensor
        data_s_train = torch.from_numpy(data_s_train)
        data_s_val = torch.from_numpy(data_s_val)
        data_t_val = torch.from_numpy(data_t_val)  # 加的验证
        data_s_test = torch.from_numpy(data_s_test)

        data_t_train = torch.from_numpy(data_t_train)
        data_t_test = torch.from_numpy(data_t_test)

        label_s_train = torch.from_numpy(label_s_train)
        label_s_val = torch.from_numpy(label_s_val)
        label_t_val = torch.from_numpy(label_t_val)  # 加的验证
        label_s_test = torch.from_numpy(label_s_test)
        # label_t_train = torch.from_numpy(label_t_train)
        label_t_test = torch.from_numpy(label_t_test)

        # seal to data-set
        train_dataset_s = Data.TensorDataset(data_s_train, label_s_train)
        train_dataset_t = Data.TensorDataset(data_t_train)
        val_dataset_s = Data.TensorDataset(data_s_val, label_s_val)
        val_dataset_t = Data.TensorDataset(data_t_val, label_t_val)  # 加的验证
        test_dataset_s = Data.TensorDataset(data_s_test, label_s_test.squeeze())
        test_dataset_t = Data.TensorDataset(data_t_test, label_t_test.squeeze())

        # print(train_dataset_s, val_dataset_s)
        criterion = nn.NLLLoss()

        train(train_dataset_s, val_dataset_s, val_dataset_t, train_dataset_t)
