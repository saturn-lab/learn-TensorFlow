# Numpy

### Numpy basics

NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。NumPy 是一个运行速度非常快的数学库，主要用于数组计算。NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引。ndarray 对象是用于存放同类型元素的多维数组，每个元素在内存中都有相同存储大小的区域，其 内部由以下内容组成：

- 一个指向数据（内存或内存映射文件中的一块数据）的指针。
- 数据类型或 dtype，描述在数组中的固定大小值的格子。
- 一个表示数组形状（shape）的元组，表示各维度大小的元组。
- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。

Numpy模块的导入：

```
import numpy
```

创建一个 ndarray 只需调用 NumPy 的 array 函数即可：

```
np.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

| object | 数组或嵌套的数列                                          |
| ------ | --------------------------------------------------------- |
| dtype  | 数组元素的数据类型，可选                                  |
| copy   | 对象是否需要复制，可选                                    |
| order  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok  | 默认返回一个与基类类型一致的数组                          |
| ndimin | 指定生成数组的最小维度                                    |

### Numpy 数据类型和数组创建

numpy 支持的数据类型比 Python 内置的类型要多很多，基本上可以和 C 语言的数据类型对应上，其中部分类型对应为 Python 内置的类型。常用 NumPy 基本类型有bool, int, float, int16, int32等。

ndarray 数组除了可以使用底层 ndarray 构造器来创建外，也可以通过以下几种方式来创建。

##### numpy.empty 

方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组：

```
numpy.empty(shape, dtype = float, order = 'C')
```

##### numpy.zeros 

创建指定大小的数组，数组元素以 0 来填充：

```
umpy.zeros(shape, dtype = float, order = 'C')
```

##### numpy.ones 

创建指定形状的数组，数组元素以 1 来填充：

```
numpy.ones(shape, dtype = None, order = 'C')
```

参数说明：

| 参数  | 描述                                                         |
| ----- | ------------------------------------------------------------ |
| shape | 数组形状                                                     |
| dtype | 数据类型，可选                                               |
| order | 有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。 |

### Numpy数组操作

#### NumPy 广播(Broadcast)

广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。如果两个数组 a 和 b 形状相同，即满足 **a.shape == b.shape**，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相同，且各维度的长度相同。

例子：

```
import numpy as np    
a = np.array([[ 0, 0, 0],[10,10,10],[20,20,20],[30,30,30]]) 
b = np.array([1,2,3]) print(a + b)
```

输出结果为：

```
[[ 1  2  3]
 [11 12 13]
 [21 22 23]
 [31 32 33]]
```

#### NumPy 形状变换

##### numpy.reshape 

该函数可以在不改变数据的条件下修改形状，格式如下： numpy.reshape(arr, newshape, order='C')

- arr：要修改形状的数组
- newshape：整数或者整数数组，新的形状应当兼容原有形状
- order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。

##### numpy.ravel

展平的数组元素，顺序通常是"C风格"，返回的是数组视图（view，有点类似 C/C++引用reference的意味），修改会影响原始数组。

该函数接收两个参数：

```
numpy.ravel(a, order='C')
```

参数说明：

- order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。

numpy.transpose 函数用于对换数组的维度，格式如下：

```
numpy.transpose(arr, axes)
```

参数说明:

- `arr`：要操作的数组
- `axes`：整数列表，对应维度，通常所有维度都会对换。

##### numpy.squeeze

numpy.squeeze 函数从给定数组的形状中删除一维的条目，函数格式如下：

```
numpy.squeeze(arr, axis)
```

参数说明：

- `arr`：输入数组
- `axis`：整数或整数元组，用于选择形状中一维条目的子集

##### numpy.concatenate

numpy.concatenate 函数用于沿指定轴连接相同形状的两个或多个数组，格式如下：

```
numpy.concatenate((a1, a2, ...), axis)
```

参数说明：

- `a1, a2, ...`：相同类型的数组
- `axis`：沿着它连接数组的轴，默认为 0

##### numpy.stack

numpy.stack 函数用于沿新轴连接数组序列，格式如下：

```
numpy.stack(arrays, axis)
```

参数说明：

- `arrays`相同形状的数组序列
- `axis`：返回数组中的轴，输入数组沿着它来堆叠

##### numpy.hstack

numpy.hstack 是 numpy.stack 函数的变体，它通过水平堆叠来生成数组。

##### numpy.vstack

numpy.vstack 是 numpy.stack 函数的变体，它通过垂直堆叠来生成数组。

### Numpy函数操作

NumPy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。

##### 三角函数

NumPy 提供了标准的三角函数：sin()、cos()、tan()。

##### 算术函数

NumPy 算术函数包含简单的加减乘除: **add()**，**subtract()**，**multiply()** 和 **divide()**。需要注意的是数组必须具有相同的形状或符合数组广播规则。

##### 统计函数

NumPy 提供了很多统计函数，用于从数组中查找最小元素，最大元素，百分位标准差和方差等。 函数说明如下：

numpy.amin() 用于计算数组中的元素沿指定轴的最小值。 numpy.amin() 用于计算数组中的元素沿指定轴的最大值。

##### NumPy 排序、条件刷选函数

numpy.sort()，numpy.sort() 函数返回输入数组的排序副本。函数格式如下：

```
numpy.sort(a, axis, kind, order)
```

##### NumPy 线性代数

NumPy 提供了线性代数函数库 **linalg**，该库包含了线性代数所需的所有功能，可以看看下面的说明：

| 函数        | 描述                             |
| ----------- | -------------------------------- |
| dot         | 两个数组的点积，即元素对应相乘。 |
| vdot        | 两个向量的点积                   |
| inner       | 两个数组的内积                   |
| matmu       | 两个数组的矩阵积                 |
| determinant | 数组的行列式                     |
| solve       | 求解线性矩阵方程                 |
| inv         | 计算矩阵的乘法逆矩阵             |

### Numpy载入与保存

Numpy 可以读写磁盘上的文本数据或二进制数据。NumPy 为 ndarray 对象引入了一个简单的文件格式：npy。

npy 文件用于存储重建 ndarray 所需的数据、图形、dtype 和其他信息。常用的 IO 函数有：

- load() 和 save() 函数是读写文件数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npy 的文件中。
- savze() 函数用于将多个数组写入文件，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npz 的文件中。
- loadtxt() 和 savetxt() 函数处理正常的文本文件(.txt 等)

### Numpy与Tersorflow的比较

#### 转换

Numpy的Ndarry可以和Tensorflow的tensor相互转化。

```
a = np.zeros((3, 3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))
```

#### 算术操作

##### 加法操作

Tensor、numpy  两个的效果一致遇到不相同的维度时，会自动扩充。但是同一维度上的大小必须一致的，除了某一维度是值是1的情况。Tensor的shape是（tensor，1）和（1,tensor）这是可以相加的，会自动扩充。

##### 矩阵乘法

Tensor : A * B 表示按元素计算
​               tf.mul(A,B)  表示按元素计算
​               tf.matmul(A,B) 表示矩阵乘法

numpy: A * B 表示按元素计算
​              dot(A,B)表示矩阵乘法

##### 数据类型转换

```
tf.to_double(a)
tf.to_float(a)
tf.cast(x, dtype, name=None)
tensor a is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2] # dtype=tf.int3212345
```

##### **形状操作**

```
1.shape
numpy：a.shape()
Tensor：a.get_shape()  tf.shape(a)

2.reshape
Tensor：tf.reshape(a, (1,4))
numpy：np.reshape(a,(1,4))

3.tf.size(a)返回数据的元素数量
tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]])    size  = 12

4.tf.rank(a) 返回tensor的rank 
#’t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor ‘t’ is [2, 2, 3]
rank(t) ==> 3

5.某一维求和
Tensor：tf.reduce_sum(b,reduction_indices=1)
numpy：np.sum(b,axis=1)12345678910111213141516171819
```

##### 数组操作

```
1.合并、连接数组
Tensor：tf.concat(0,[a,b])第一个参数表述位数，若a （1，128，128，3）  b( 1，128，128，3）
tf.concat(0,[a,b])  ( 2，128，128，3）
numpy：vstack 和 hstack  stack(a,axis=)

2.获取整行整列数据
Tensor：
temp = tf.constant(0,shape=[5,5])
temp1 = temp[0,:] 获取某行
temp2 = temp[:,1] 获取某列
temp[1,1]  获取某个元素
temp[1:3,1:3]  获取某个范围的行列元素 

3.打包
tf.pack(values, axis=0, name=’pack’)
# ‘x’ is [1, 4], ‘y’ is [2, 5], ‘z’ is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]] 
# 沿着第一维pack
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
等价于tf.pack([x, y, z]) = np.asarray([x, y, z])

4.tf.transpose(a, perm=None, name=’transpose’)
调换tensor的维度顺序
```

##### **矩阵相关操作**

```1.tf.matrix_inverse  方阵的逆矩阵  
1.tf.matrix_inverse  方阵的逆矩阵
2.tf.matrix_determinant  方阵的行列式
3.tf.transpose转置  
4.tf.diag  给定对角线上的值，返回对角tensor1234```
```

#### 参考资料

https://docs.scipy.org/doc/numpy/user/quickstart.html

http://www.runoob.com/numpy/numpy-tutorial.html

https://blog.csdn.net/csdn15698845876/article/details/73380803

https://blog.csdn.net/jiandanjinxin/article/details/78084390?utm_source=blogxgwz0
