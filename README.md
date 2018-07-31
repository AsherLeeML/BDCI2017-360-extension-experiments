# BDCI2017-360-extension-experiments

前段时间忙里偷闲，参考第三名大佬 https://github.com/fuliucansheng/360 的代码，重做了一部分实验，结果可以看几个 Notebook 里的输出记录。非常有意思。

为了节省时间，训练数据随机抽了初赛 5w、复赛 6w，也就是各取十分之一后合成 11w 条使用。初赛复赛样本比例基本没有变化，所以没做其他处理直接拿来用了。

**想法有这样几条：** 

- 只取原数据集的一部分，用一模一样的代码，能达到什么样的分数？
- 不同模型的训练时间/效率和分数怎么样？
- 另外 Google 有个非官方的 SentencePiece 包，用 data-driven 的方法在诸如中文、日文等不用空格分隔词汇的语料上得到所谓“亚词单元”（sub-word units），类似中文 NLP 中的分词工具。想试试这玩意儿好不好用。
- 一直没搞懂 Keras 里 Timedistributed 封装器是干啥的，想测一下看看有什么作用。

还是时间关系，我只复现了 TextCNN、HAN 和 HCNN （其实就是带 Attention 的 RCNN）仨模型。

目录基本没动，只不过把关键代码放到了 Notebook 里方便直接看结果。

WVutils 里存了 Word2Vec 的代码，train 里的是训练过程。数据比较大就都删掉了，没有上传。



**实验结果有这么几条：** 

- 训练数据规模的确很重要。11w 样本最高的 f1-score 是 0.76 左右，到这个数就开始过拟合了。而开源大佬的成绩是0.90+，距离还挺大。
- 不过反过来说，只用了十分之一的数据就达到了0.76，已经相当可以了。
- TextCNN 在保证最高分数的情况下，训练速度是最快的。不过也是理所应当，作者的各模型版本中 TextCNN 只有卷积 block，而 HAN 有两个 GRU-Attention 模组，HCNN 是卷积 block + GRU-Attention 模组。
- 让人意外的是这俩模型不光训练得慢分数还低……
- 比较失望 SentencePiece 的结果比 Jieba 分词还是差了一点，也可能是不会调教。我直接用官方 API 做的……



**速度对比：** 

- TextCNN 是 190s 一个 epoch，最高 F1 为 0.76
- HAN 是 310s 一个 epoch，最高 F1 为 0.70 
- HCNN 是 295s 一个 epoch，最高 F1 为 0.69



非常奇妙，去掉 Timedistributed 这个 wrapper 后结果好像没什么变化，训练速度还快了大概十五秒。那个模型我没有继续训练，但看样子 F1 应该还能更低一些，所以结果应当与有这玩意儿的情况非常接近。

HAN 和 HCNN 既没有 TextCNN 快精度也没有更高，比较遗憾。不过这倒跟鹅厂大佬 https://github.com/brightmart/text_classification 在知乎数据集上的结果很接近。并且 HAN 和 HCNN 比 TextCNN 更快进入了过拟合。

