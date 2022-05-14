#!/usr/bin/env python
# coding: utf-8

# In[1]:


####data_utils.py
import os
import json
import pickle
import random

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def downsample(img):
    img2=np.array(img.resize((256,256), Image.BICUBIC))
    subimg=img2[::8,::8]
    #subimg=subimg/255.

    return subimg
def plot_class_preds(net,
                     images_dir: str,
                     transform,
                     num_plot: int = 5,
                     device="cuda"):
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    label_path = os.path.join(images_dir, "label.txt")
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    #json_label_path = './class_indices.json'
    #assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    #json_file = open(json_label_path, 'r')
    # {"0": "daisy"}
    #flower_class = json.load(json_file)
    # {"daisy": "0"}
    #class_indices = dict((v, k) for k, v in flower_class.items())
    class_indices = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    label_indices = dict((v, k) for k, v in class_indices.items())
    # reading label.txt file
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert len(split_info) == 3, "label format error, expect file_name and class_name"
                image_name, class_name,class_label = split_info
                image_path = os.path.join(images_dir, image_name)
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in label_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name,class_label])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        pass
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []
    labels = []
    origin_images=[]
    class_indices = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                     5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    for img_path, class_name,class_label in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        origin_images.append(img)
        label_index = int(class_label)

        # preprocessing
        img=downsample(img)
        #plt.imshow(img)
        #plt.axis('off')
        #plt.title("{}".format(class_indices[label_index]))
        #plt.show()
        #print(img.shape)
        img = transform(img)


        images.append(img)

        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
    for i in range(num_imgs):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(1, num_imgs, i+1, xticks=[], yticks=[])

        # CHW -> HWC
        #npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        #npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        plt.imshow(origin_images[i])

        title = "{}, {:.2f}%\n(label: {})".format(
            class_indices[int(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            class_indices[int(labels[i])] # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

    return fig


# In[6]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='baseline'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_baseline"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_baseline/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=False #是否显示增强过的数据图片
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_baseline/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_baseline/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_baseline/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,
                                               num_workers=num_thread)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_baseline/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[7]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='mixup'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_mixup"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_mixup/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=False #是否显示增强过的数据图片
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_mixup/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_mixup/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_mixup/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,
                                               num_workers=num_thread)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_mixup/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[8]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='cutout'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_cutout"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_cutout/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=False #是否显示增强过的数据图片
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_cutout/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_cutout/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_cutout/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,
                                               num_workers=num_thread)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_cutout/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[9]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='cutmix'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_cutmix"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_cutmix/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=False #是否显示增强过的数据图片
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_cutmix/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_cutmix/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_cutmix/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,
                                               num_workers=num_thread)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_cutmix/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[2]:


#############resmodel.py 

'''ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

def test_resnet():
    net = ResNet50()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_resnet()


# In[7]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=ResNet18(num_classes=100).to(device)
net


# In[3]:


##########train_eval_utils.p

import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

class_indices={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


def train_one_epoch_data_augmentation(model, optimizer, data_loader, device, epoch,method='baseline',
                                      argbeta=1.0,prob=-1.0,n_holes=1,length=16,picshow=False):
    if method not in ['baseline','mixup','cutout','cutmix']:
        print('method error!')
    if prob<0 or prob>1:
        problist={'baseline':1.0,'mixup':1.0,'cutout':1.0,'cutmix':0.5}
        prob=problist[method]
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        input=images.to(device)
        target=labels.to(device)
        target_a = target
        target_b = target
        lam=1.0

        r=np.random.rand(1)
        if method!='baseline' and (argbeta>0 and r<prob):
            if method=='cutout':
                _, _, h, w = input.shape
                h = input.shape[2]
                w = input.shape[3]
                lam = 1 - (length ** 2 / (h * w))
                for _ in range(n_holes):
                    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                    input[:, :, bbx1:bbx2, bby1:bby2] = 0.
            else:
                lam = np.random.beta(argbeta, argbeta)
                rand_index = torch.randperm(input.size()[0]).to(device)
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                if method=='cutmix':
                    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                else:#method=mixup
                    input = lam * input + (1 - lam) * input[rand_index, :, :]
        output = model(input)
        loss = loss_function(output, target_a) * lam + loss_function(output, target_b) * (1. - lam)

        if picshow:
            num_imgs = 5
            fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
            for numkk in range(num_imgs):
                ax = fig.add_subplot(1, num_imgs, numkk + 1, xticks=[], yticks=[])
                img = input[numkk].cpu().numpy().transpose(1, 2, 0)
                img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                if method=='baseline' :
                    title = "{}\nlabel:{}".format(method, class_indices[int(target[numkk].cpu().numpy())])
                elif method=='cutout':
                    title = "{}\nlabel:{}({})".format(method,class_indices[int(target[numkk].cpu().numpy())],np.round(lam,2))
                else:
                    title = "{}\nlabel:{}({})\nadd label:{}({})".format(method,
                        class_indices[int(target_a[numkk].cpu().numpy())], np.round(lam,2),
                        class_indices[int(target_b[numkk].cpu().numpy())], np.round(1 - lam,2))
                ax.set_title(title)
                plt.axis('off')
                plt.imshow(img.astype('uint8'))
            plt.show()


        preds = torch.max(output, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()
        loss.backward()
        # mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        total_loss+=loss.detach()
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(total_loss.item()/(step+1), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    # mean_loss=mean_loss*step/num_samples
    mean_loss=total_loss/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        preds = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()

        # 计算预测正确的比例

        # print(pred.shape,labels.to(device).shape,'test')
        loss = loss_fun9ction(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    mean_loss=mean_loss*step/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

def train_one_epoch_mixup(model, optimizer, data_loader, device, epoch,argbeta=1.0,prob=0.5):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data

        input=images.to(device)
        target=labels.to(device)

        r=np.random.rand(1)
        # print(input.shape,'shape')
        if argbeta>0 and r<prob:
            lam= np.random.beta(argbeta,argbeta)
            rand_index=torch.randperm(input.size()[0]).to(device)
            target_a=target
            target_b=target[rand_index]
            # rand_index=torch.randperm(input.size()[0]).to(device)
            input=lam*input+(1-lam)*input[rand_index,:,:]
            output=model(input)
            loss=loss_function(output,target_a)*lam+loss_function(output,target_b)*(1.-lam)

            numkk = 1
            img = input[numkk].cpu().transpose(1, 2, 0)
            img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
            # print(img,'img')
            plt.imshow(img.astype('uint8'))
            title = "mixup\nlabel:{}({}),{}({})".format(
                    class_indices[int(target_a[numkk].cpu().numpy())],lam,
                    class_indices[int(target_b[numkk].cpu().numpy())],1-lam,
            )
            plt.title(title)
            plt.axis('off')
            plt.imshow(img.astype('uint8'))
            # print(target[numkk], output[numkk],'no')
            plt.show()


        else:
            output=model(input)
            # output = torch.max(model(input), dim=1)[1]
            # print( target.shape, output.shape, 'shape')
            loss=loss_function(output,target)


        # pred = model(images.to(device))
        preds = torch.max(output, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()

        # 计算预测正确的比例


        #loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    mean_loss=mean_loss*step/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

def train_one_epoch_cutout(model, optimizer, data_loader, device, epoch,n_holes=1,length=16,prob=0.5):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data

        input=images.to(device)
        target=labels.to(device)

        r=np.random.rand(1)
        if  r<prob:
            _,_,h,w=input.shape
            # print(input.shape)

            h=input.shape[2]
            w=input.shape[3]
            lam=1-(length**2/(h*w))
            for _ in range(n_holes):
                bbx1,bby1,bbx2,bby2=rand_bbox(input.size(),lam)
                input[:,:,bbx1:bbx2,bby1:bby2]=0.

        output=model(input)
        # output = torch.max(model(input), dim=1)[1]
        # print( target.shape, output.shape, 'shape')
        loss=loss_function(output,target)

        # numkk = 1
        # img = input[numkk].numpy().transpose(1, 2, 0)
        # img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
        # print(img,'img')
        # plt.imshow(img.astype('uint8'))
        # title = "cutout\nlabel:{}".format(class_indices[int(target[numkk].numpy())])
        # plt.title(title)
        # plt.axis('off')
        # plt.imshow(img.astype('uint8'))
        # # print(target[numkk], output[numkk],'no')
        # plt.show()

        # pred = model(images.to(device))
        preds = torch.max(output, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()

        # 计算预测正确的比例


        #loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    mean_loss=mean_loss*step/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

def train_one_epoch_cutmix(model, optimizer, data_loader, device, epoch,argbeta=1.0,prob=0.5):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        # numkk = 1
        # img = images[numkk].numpy().transpose(1, 2, 0)
        # img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
        # # print(img, 'img')
        # plt.imshow(img.astype('uint8'))
        # print('orign')
        # # print(target[numkk], output[numkk], 'no')
        # plt.show()

        input=images.to(device)
        target=labels.to(device)

        r=np.random.rand(1)
        print(input.shape,'shape')
        if argbeta>0 and r<prob:
            lam= np.random.beta(argbeta,argbeta)
            rand_index=torch.randperm(input.size()[0]).to(device)
            target_a=target
            target_b=target[rand_index]
            bbx1,bby1,bbx2,bby2=rand_bbox(input.size(),lam)
            input[:,:,bbx1:bbx2,bby1:bby2]=input[rand_index,:,bbx1:bbx2,bby1:bby2]
            lam=1-((bbx2-bbx1)*(bby2-bby1)/(input.size()[-1]*input.size()[-2]))
            output=model(input)
            # output = torch.max(model(input), dim=1)[1]
            # print(target_a.shape,target_b.shape,target.shape,output.shape,'shape')
            loss=loss_function(output,target_a)*lam+loss_function(output,target_b)*(1.-lam)
            # numkk=1
            # img = input[numkk].numpy().transpose(1, 2, 0)
            # img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
            # plt.imshow(img.astype('uint8'))
            # print(target_a[numkk],target_b[numkk],output[numkk],lam.item())
            # plt.show()
        else:
            output=model(input)
            # output = torch.max(model(input), dim=1)[1]
            # print( target.shape, output.shape, 'shape')
            loss=loss_function(output,target)

            # numkk = 1
            # img = input[numkk].numpy().transpose(1, 2, 0)
            # img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
            # print(img,'img')
            # plt.imshow(img.astype('uint8'))
            # print(target[numkk], output[numkk],'no')
            # plt.show()

        # pred = model(images.to(device))
        preds = torch.max(output, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()

        # 计算预测正确的比例


        #loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    mean_loss=mean_loss*step/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    score_list = []
    label_list = []
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        # pred=nn.Softmax(pred,dim=1)
        loss+=loss_function(pred,labels.to(device))
        m = nn.Softmax(dim=1)
        pred = m(pred)
        #print(perloss)
        #loss+=perloss
        pred,predlabel = torch.max(pred, dim=1)
        score_list.extend(pred.detach().cpu().numpy())
        label_right = torch.eq(predlabel, labels.to(device))
        # print(label_right)
        # print(preds)
        label_list.extend(label_right.detach().cpu().numpy())
        #l=torch.eq(pred, labels.to(device))
        #print(l)
        #print(images[l])
        #print(pred[l])
        #print(labels[l])

        sum_num += torch.eq(predlabel, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    # print(num_samples)
    loss = loss / num_samples
    score_array = np.array(score_list)
    label_array = np.array(label_list)

    return loss.item(),acc,score_array,label_array

@torch.no_grad()
def evaluate1(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss+=loss_function(pred,labels.to(device))
        #print(perloss)
        #loss+=perloss
        probs, pred = torch.max(torch.softmax(pred, dim=1), dim=1)
        #probs = torch.max(pred, dim=1)[0]
        #pred = torch.max(pred, dim=1)[1]

        l=torch.eq(pred, labels.to(device))
        l=[not i for i in l]
        falseimg=images[l]
        falselabel=labels[l ]
        falsepred=pred[l]
        class_indices={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        # import matplotlib.pyplot as plt
        num_imgs=10
        fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
        for i in range(num_imgs):
            # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
            ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

            # CHW -> HWC
            # npimg = images[i].cpu().numpy().transpose(1, 2, 0)

            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            # npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img=falseimg[i].cpu().numpy().transpose(1,2,0)
            img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255


            title = "{}\n{:.2f}%\n{}".format(
                class_indices[int(falsepred[i])],  # predict class
                probs[i] * 100,  # predict probability
                class_indices[int(falselabel[i])]  # true class
            )
            ax.set_title(title)
            plt.imshow(img.astype('uint8'))
            plt.axis('off')
        plt.show()
       #plt.axis('off')
        #plt.show(fig)

        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    loss = loss / num_samples

    return loss.item(),acc


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# In[10]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='cutmix'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_cutmix1"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_cutmix1/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=True #是否显示增强过的数据图片
    # 超参数设置
    epochs = 5  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_cutmix1/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_cutmix1/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_cutmix1/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,
                                               num_workers=num_thread)  

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_cutmix1/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[11]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='mixup'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_mixup1"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_mixup1/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=True #是否显示增强过的数据图片
    # 超参数设置
    epochs = 5  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_mixup1/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_mixup1/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_mixup1/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,
                                               num_workers=num_thread)  
    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_mixup1/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[9]:


#######cifrapytorch.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

#from data_utils import plot_class_preds
#from train_eval_utils import *

#from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='cutout'#baseline,mixup,cutout,cutmix
    model_path = "E:/originnet_cutout1"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = 'E:/originnet_cutout1/weights/cutout/model-99.pth'  # 模型加载权重的路径
    
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    
    #pic="C:/Users/94313/Desktop/originnet3/result_img"
    #if not os.path.exists(pic):
        #os.mkdir(pic)
    
    ispcishow=True #是否显示增强过的数据图片
    # 超参数设置
    epochs = 5  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='E:/originnet_cutout1/weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="E:/originnet_cutout1/logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="E:/originnet_cutout1/logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,
                                               num_workers=num_thread)  

    testset = torchvision.datasets.CIFAR10(root='C:/Users/94313/Desktop/data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")
 
    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="E:/originnet_cutout1/result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d | %.08f| %.08f|'
                    '%.8f |%.8f |%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()


# In[ ]:




