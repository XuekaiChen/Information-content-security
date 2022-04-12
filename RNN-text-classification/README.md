# 基于Pytorch的LSTM/RNN中文文本分类
数据集可在[此处](https://pan.baidu.com/s/1UQ3fMOHTG0ztuSd8PEaNZg)下载获得， 
提取码：`i4o2`，解压后放置于项目目录下。文件格式：.txt文本文件

数据格式如下：
```
6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
```
每行为一条数据，以`_!_`分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词

分类code与名称：

```
class2id = {
    'news_story': 0,  # 民生 故事
    'news_culture': 1,  # 文化 文化
    'news_entertainment': 2,  # 娱乐 娱乐
    'news_sports': 3,   # 体育 体育
    'news_finance': 4,  # 财经 财经
    'news_house': 5,    # 房产 房产
    'news_car': 6,  # 汽车 汽车
    'news_edu': 7,  # 教育 教育
    'news_tech': 8,   # 科技 科技
    'news_military': 9,   # 军事 军事
    'news_travel': 10,    # 旅游 旅游
    'news_world': 11,   # 国际 国际
    'stock': 12,    # 证券 股票
    'news_agriculture': 13,   # 农业 三农
    'news_game': 14   # 电竞 游戏
}
```
数据规模：
共382688条，分布于15个分类中。

## 1.数据数据预处理
为避免调试程序时重复处理，可先运行`data_process.py`进行数据预处理，包含对数据的分词、标签化等操作。

## 2.模型定义
可运行`model.py`文件模拟模型forward过程，可自行调节使用LSTM或RNN模块。

## 3.训练/测试
运行`main.py`开始模型训练，或输入已有**模型参数文件**（注意，不是模型文件）的路径，按程序提示进行加载。

## 4.查看loss曲线与accuracy曲线
可以看到，同样的参数设置下，RNN与LSTM的模型效果差距还是很大的。

![loss曲线](image/loss.png)

![accuracy曲线](image/accuracy.png)
