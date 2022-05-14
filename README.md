# midterm
task 1 分为三部分来做，主要实现代码，作图代码和图像显示代码。

task 2
数据集的准备
使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录

数据集的处理
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。

开始网络训练
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。

训练结果预测
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。我们首先需要去frcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。
model_path指向训练好的权值文件，在logs文件夹里。
classes_path指向检测类别所对应的txt。
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。检测模式有很多，在predict.py内有详细说明。


运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

运行get_proposal.py即可获得fasterrcnn第一阶段的候选框，测试前，将图片放在imgs内，输出图片会在proposal_out
