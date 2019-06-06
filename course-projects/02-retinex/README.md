也是上上周布置的作业，主要是比较不同 Retinex 算法实现的结果。同样也是需要自己看论文并实现算法，这点应该是选这门课最大的优点了，也是硕士需要掌握的基本技能。

今天在课上，还以为会被批评，没想到被夸奖了一翻，心里美滋滋。授课教授说我写的 report 很清晰明了，可以清晰地知道哪张结果图片对应哪个算法，还问我是不是写过很多论文。哈哈，论文是没写过的，倒是看过不少，知道非英语母语读者的痛点在哪里以及一些基本的套路。


### Retinex 理论
Retinex 这个词由 Retina 和 Cortex 两个单词组成。在 Retinex 理论中，物体的颜色是由物体对长波、中波和短波光线的反射能力决定的，而不是由反射光强度的绝对值决定的，并且物体的色彩不受光照非均性的影响，具有一致性。

![](http://upload-images.jianshu.io/upload_images/2759738-a63ad1949cff450d.png&originHeight=368&originWidth=576&size=30403&status=done&width=461?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 Retinex 理论中，人眼得到的图像数据取决于入射光和物体表面对入射光的反射。如上图所示，I(x,y) 是我们最终得到的图像数据，先是由入射光照射，然后经由物体反射进入成像系统，最终形成我们所看到的图像。该过程可以用公式表示：

![](https://upload-images.jianshu.io/upload_images/2759738-2caa132b9a9041a8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


其中，I(x,y)代表被观察或照相机接收到的图像信号；L(x,y) 代表环境光的照射分量 ；R(x,y) 表示携带图像细节信息的目标物体的反射分量。

将该式子两边取对数，可以得到物体原本的信息:

![](https://upload-images.jianshu.io/upload_images/2759738-f12d90242c4c13b3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在图像处理领域，常将该理论用于图像增强，为了得到成像更好的图片。这时，R(x,y) 表示为图像增强得到后的图像，I(x,y) 为原始的图像。在处理过程中 L(x,y) 常为 I(x,y) 高通滤波之后的结果，也可以用其他滤波的方法，比如中值滤波，均值滤波等等。

### SSR 算法
SSR (Singal Scale Retinex)，即单尺度视网膜算法是 Retinex 算法中最基础的一个算法。运用的就是上面的方法，具体步骤如下：

* 输入原始图像 I(x,y) 和滤波的半径范围 sigma;
* 计算原始图像 I(x,y) 高斯滤波后的结果，得到 L(x,y);
* 按照公式计算，得到 Log[R(x,y)]；
* 将得到的结果量化为 [0, 255] 范围的像素值，然后输出结果图像。

需要注意的是，最后一步量化的过程中，并不是将 Log[R(x,y)] 进行 Exp 化得到 R(x,y) 的结果，而是直接将 Log[R(x,y)] 的结果直接用如下公式进行量化：

![](https://upload-images.jianshu.io/upload_images/2759738-fb740bb3aef91eaa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将过程整合在一起就是如下过程：

![](http://upload-images.jianshu.io/upload_images/2759738-9f42d0976c61817c.png&originHeight=88&originWidth=719&size=16583&status=done&width=575?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### MSR 算法
MSR (Multi-Scale Retinex)，即多尺度视网膜算法是在 SSR 算法的基础上提出的，采用多个不同的 sigma 值，然后将最后得到的不同结果进行加权取值，公式如下所示：

![](http://upload-images.jianshu.io/upload_images/2759738-270f2ee015a143ca.png&originHeight=91&originWidth=463&size=11926&status=done&width=370?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 `n`  是尺度的数量， `σ= {σ1，σ2，...，σn}` 是高斯模糊系数的向量， `wk` 是与第 `k` 个尺度相关的权重，其中 `w1 + w2 + ... + wn = 1` 。


### MSRCR 算法
MSRCR 算法是一种改进 MSR 的算法，全称是 Multi-Scale Retinex with Color Restoration，即带色彩恢复的多尺度视网膜增强算法。

![](http://upload-images.jianshu.io/upload_images/2759738-9270f3311955729f.png&originHeight=152&originWidth=711&size=22150&status=done&width=569?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

就是在 MSR 的基础上，加上了色彩恢复的功能。详细的内容及公式可以查看论文 《A multiscale retinex for bridging the gap between color images and the human observation of scenes》。

### 作业要求


#### 任务 1
在论文中随机选择两个 MSR 算法，实现并比较。

#### 任务 2

* sigma = 16,32,64 时，比较不同高斯滤波的性能
* 1) 读取并对图像进行二维高斯滤波：
  * input image - X; 高斯滤波后的图片结果 - Y; 令 Z = X/Y
  * 求 Z 的均值（mean）和标准差（sigma）
  * 以均值为中心，取 -2*sigma 到 2*sigma 范围内的值，然后将其扩展到 [0,255] 范围并输出图像
* 2) 读取并对图像进行二维高斯滤波：
  * input image - X; 高斯滤波后的图片结果 - Y; 令 Z = logX - logY
  * 求 Z 的均值（mean）和标准差（sigma）
  * 以均值为中心，取 -2*sigma 到 2*sigma 范围内的值，然后将其扩展到 [0,255] 范围并输出图像


### 结果比较


#### 一些标注
在比较结果之前，先对一些标注信息进行说明，这样会有助于后续的结果对比。

* _SSR : Single Scale Retinex_

* _SSR - DIV : Single Scale Retinet ( Z = X/Y )_

* _SSR - LOG : Single Scale Retinex (logZ = logX – logY)_

* _MSR : Multi - Scale Retinex_

* _MSRCR : Multi - Scale Retinex with Color Restoration_


#### 实验结果

![](http://upload-images.jianshu.io/upload_images/2759738-8ee43c15a26caf81.png&originHeight=1145&originWidth=956&size=810332&status=done&width=746?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/2759738-deb34d8a315e8d4e.png&originHeight=1179&originWidth=992&size=942650&status=done&width=746?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/2759738-96339a0fd83dc997.png&originHeight=1185&originWidth=1011&size=941687&status=done&width=746?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/2759738-362f4c3215f5ae3f.png&originHeight=1363&originWidth=957&size=873857&status=done&width=746?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 不同 sigma 取值的对比
在 SSR 图像组、SSR- DIV 图像组和 SSR- LOG 图像组中，我们可以看到当 sigma 值不大的时候(16-128)，增强后的图像亮度比原图像要暗。而且图像的亮度随着 sigma 值的增加而增加。当 sigma=256 时，图像的亮度将与原始图像相似。


#### 不同 SSR 算法对比
通过比较不同的 SSR 方法，可以发现当 sigma 值在 16-128 之间时，SSR-DIV 的结果亮度最暗，而 SSR-LOG 的结果稍暗，但结果比 SSR 的结果更蓝。对于 SSR 和 SSR-LOG 算法，sigma 选择 64 或 128 是当前实验图片的最佳结果。对于 SSR-DIV 的结果，选择 256 是最佳选择。


#### 不同 MSR 算法对比
通过对 MSR 算法和 SSR 算法的比较，可以发现，MSR 和 MSRCR 算法的结果一般要比 SSR 算法的结果更亮，因为多重尺度（多个 sigma 的取值）的组合。对两种 MSR 算法的结果进行比较，可以发现，结果没有太大的差别。而且在这些组合中，64-128-256 的组合结果是最优的。不仅图像变得更清晰，而且颜色也变得更加明亮。此外，对于 MSRCR 算法来说，太多的参数是一个负担（*实验中只采用了论文推荐的经验参数，而没有尝试更多的参数比较*）。


### 参考

[1]. [http://www.cnblogs.com/Imageshop/archive/2013/04/17/3026881.html](http://www.cnblogs.com/Imageshop/archive/2013/04/17/3026881.html)

[2]. [http://www.cnblogs.com/Imageshop/p/3810402.html](http://www.cnblogs.com/Imageshop/p/3810402.html)
