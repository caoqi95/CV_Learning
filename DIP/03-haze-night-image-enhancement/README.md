数字图片处理的第 3 个项目，一下子优化两种图片类型，而且紧接着后面又来了项目 4。断断续续，差不多一星期，看论文，看别人的代码，实现完成了。

### 涉及的理论及公式

可以查看这篇文章：[论文记录 - Single Image Haze Removal Using Dark Channel Prior]([https://www.jianshu.com/p/ed377aaaf8cf](https://www.jianshu.com/p/ed377aaaf8cf)
)

下面是一些重要的公式：

![](https://upload-images.jianshu.io/upload_images/2759738-5d06464ee682b171.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此时，假设 `A` 是给定的，具体的 `A` 的取值会在后面说明。接下来对公式 (1) 进行整理转换可以得到：

![](https://upload-images.jianshu.io/upload_images/2759738-df9b5288e578ec48.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

需要注意的是，该公式是针对每个颜色通道的，所以用 I<sup>c</sup> 表示。

然后再假设 Ω(x) 是一个常数，并将 t(x) 用 t&#772;(x) 来表示。然后在公式 (7) 两边计算暗通道，最后两边进行最小值操作：

![](https://upload-images.jianshu.io/upload_images/2759738-7c2e8f97878ad5fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为 t&#772;(x) 是常数，所以可以将其提取出来。

因为 J 为无灰度图像，即待求的图像，根据之前的暗通道理论，J 的暗通道接近于零:

![](https://upload-images.jianshu.io/upload_images/2759738-bfb4c7844f57f4c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

又因为 A<sup>c</sup> 总是为正，所以有：

![](https://upload-images.jianshu.io/upload_images/2759738-b31e12a3879d4b78.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将公式 (10) 代入到 公式 (8)，可以得到：

![](https://upload-images.jianshu.io/upload_images/2759738-000715bde53a9609.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样就可以根据已知的 `I` 和给定的 `A` 来求得 `t(x)`，继而就可以求得 `J` 了。
> In practice, even on clear days the atmosphere is not absolutely free of any particle. So the haze still exists when we look at distant objects. Moreover, the presence of haze is a fundamental cue for human to perceive depth [13], [14]. This phenomenon is called aerial perspective. If we remove the haze thoroughly, the image may seem unnatural and we may lose the feeling of depth.

实际上，即使在晴朗的日子，大气中也并非完全没有任何粒子。所以当我们看远处的物体时，雾气仍然存在。此外，雾气的存在是人类感知深度的基本线索。这种现象被称为空中透视。如果彻底去除雾气，图像反而可能会看起来不自然，而且也会有失去深度的感觉。

所以在公式 (11) 中加入一个范围在 [0, 1] 的因子 ω：

![](https://upload-images.jianshu.io/upload_images/2759738-67878b27b2798a3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在论文的后半部分，也讨论了关于 `A` 的取值问题，因为上面假定了 `A` 是给定的。作者认为先前的工作，很少将注意力放在 most haze-opaque 区域，即最模糊不透明的地方。在一篇论文中，提出将亮度值最大的像素认为是最模糊不透明的区域，但这只在天气为阴天，阳光可以忽略的情况下成立。但是，在实际中我们不能忽略阳光。

之后，作者提出使用暗通道来检测最模糊不透明的区域来提升 `A` 值的评估。方法如下：

+ 从暗通道中取亮度值为前 0.1% 的像素；
+ 基于这些像素，在原始图像中寻找其对应的具有最高亮度的像素值作为 `A` 的值。

最后，利用公式 (1) 恢复 `J` 的时候，当 `t(x)` 趋近于 0 的时候，会导致 `J` 值异常大，会易于产生噪音。因此，对 `t(x)` 加入一个下边界 `t0`，最后的恢复公式如下所示：

![](https://upload-images.jianshu.io/upload_images/2759738-96a733e4870a70ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

`t0` 的取值常为 0.1。


### 去雾优化结果

在这个项目中，`A` 的取值我只取了所有通道的均值，这个与论文不同。同时在利用暗通道得到处理后的结果会有点粗糙，如下图所示。论文中使用 *Soft Mapping* 来获得更细腻的结果。但是普遍认为 *Soft Mapping* 算法比较复杂且效率低，所以在项目中用了何恺明的另一篇论文的算法 - [*Guided Filtering*](http://kaiminghe.com/eccv10/) 来得到更好的处理结果。

![](https://upload-images.jianshu.io/upload_images/2759738-69aa2e3e3e453009.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面是一些处理结果展示：

![](https://upload-images.jianshu.io/upload_images/2759738-900a62507a78cdf3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-7e0dff4758f3d227.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-5039ddb10b73afc6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-76cd1366e1b90683.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以发现，处理的结果，图像都偏蓝或者偏深色。这和 `A` 的取值有关，可以在取 `A` 值的时候，对其上限进行一定的设置。

### 夜间图像增强结果
[Fast Efficient Algorithm For Enhancement Of Low Lighting Video](https://ieeexplore.ieee.org/document/6012107) 这篇论文基于暗通道理论进行了夜间图像增强的研究。增强的结果很好，如下图所示：

![](https://upload-images.jianshu.io/upload_images/2759738-008f9ce8d3498350.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但我用该论文中的算法来对教授提供的一些图片进行增强处理，发现结果并没有论文中那么好的结果。

可以原因有：

+ 图片的原因：论文中使用的是几乎完全黑的图片，如上面的图片所示。而教授提供的图片则会有一些亮光，并非完全黑。
+ 我算法实现的问题，鉴于时间原因（急着交作业），没有时间细究论文和代码。

于是，报着试一试的想法，直接用上面去雾的暗通道算法来对夜间图像进行处理，发现增强的结果意外地好，结果如下所示：

![](https://upload-images.jianshu.io/upload_images/2759738-641d74ff2d0bc908.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-d61d9fe156f723ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-ff54b130f9681afe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-5084d8b3df45cbb9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同样在处理结果上使用 *Guided Filter*，会带来更优的效果。 