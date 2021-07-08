# 毕业论文 Artifacts Evaluation

## 1.概述

```
此仓库用于我的毕业论文的工件评估，论文题目为“超大规模分类网络中全连接层的显存分配问题”。
论文链接<https://docs.qq.com/pdf/DYWRkelpweWlqRmVp>
代码主要内容为一系列适用于指定类型的大规模矩阵乘法的分块接口。接口通过调用cuBLAS库中的相关函数实现，运行在CUDA平台。
```

## 2.仓库内容介绍

* 代码

  论文相关的代码可直接从本仓库访问得到。

* 技术文档

  用于介绍环境，指导代码的编译、运行、测试和使用。

## 3.代码内容介绍

本仓库提供的分块接口适用于两种矩阵乘类型（具体阐述参见论文），类型与分块逻辑如下图所示：

### GEBP型矩阵乘

![image-20210706101309860](/Users/caowanlu/Library/Application Support/typora-user-images/image-20210706101309860.png)

### GEPDOT型矩阵乘

![image-20210706101427144](/Users/caowanlu/Library/Application Support/typora-user-images/image-20210706101427144.png)

### 分块接口介绍

论文中主要讨论了四种分块接口：Split_GEBP()，Split_GEBP_Mul()，Split_GEPDOT()和Split_GEPDOT_Mul()，分别对应GEBP和GEPDOT型矩阵乘分块算法的单流实现和多流并行实现。文中也分析了四种接口的时间表现，最终保留了Split_GEBP_Mul()，Split_GEPDOT()和Split_GEPDOT_Mul()。

运行效果如下：

![image-20210706102052940](/Users/caowanlu/Library/Application Support/typora-user-images/image-20210706102052940.png)

![image-20210706102151973](/Users/caowanlu/Library/Application Support/typora-user-images/image-20210706102151973.png)

## 4.环境准备与运行

**Step 0: 运行环境介绍**

Ubuntu 18.04.5 LTS，cuda 11.0

GPU型号：TITAN RTX

**Step 1: 编译&Getting Started**

clone代码到本地，在主目录下

```
//编译
$ make sp

//运行可执行文件split即可完成规模为m*k和k*n的矩阵乘计算，需要输入参数m,n,k和接口选择，格式如下：
//p.s. funcname可取“gepdot”，“gepdot_mul”或“gebp_mul”，分别与上文介绍的三个函数对应
$ ./split m k n funcname

//Example：
$ ./split 1024 1024 1024 gepdot
$ ./split 1024 1024 1024 gepdot_mul
$ ./split 1024 1024 1024 gebp_mul
```

**Step 2: 数据复现与后续使用**

文中数据复现：

1. 结合step1中介绍，按论文中对应参数调用接口，在相同的环境中运行，理论上就能复现文中数据。
2. `test.sh`中也准备了可能用到的测试数据，可以直接运行test.sh进行测试

直接使用分块接口：

如果需要在您的代码中直接调用本库中提到的分块函数，请复制`common.h`和`common.cu`到您的合适的目录中，并参考其定义与参数格式加以使用。

