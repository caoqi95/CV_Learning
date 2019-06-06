上上上上周，数字图片处理课程布置了一个作业，需要看论文实现并比较各种直方图均衡的算法：

![](https://upload-images.jianshu.io/upload_images/2759738-a2a628f409acdef6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

基本的直方图均衡算法已经在[这篇](https://www.jianshu.com/p/726b7b284ef3)文章里说明了，今天这篇主要讲变体。

最近忙于课业和准备自己的课题，这篇躺在草稿箱里很久了，今天提前做完作业，才有时间整理出来。

### Histogram Equalization 的缺陷

如下面的图片所示，可以看出，原图片在直方图均衡之后亮度变的异常大，这样给人的感觉会很不自然。

![](https://upload-images.jianshu.io/upload_images/2759738-b0ff098408625e2c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

再看下面的图片，原图中的云朵和均衡之后的云朵完全是不一样的视觉感受，均衡之后的云朵都是乌云与原图相差较大。而且飞机尾部的字母标志和标志周围的对比度也降低了，几乎看不清 *F-16* 的标志。

![](https://upload-images.jianshu.io/upload_images/2759738-de7f9e6e68367a95.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 [Bi-Histogram Equalization](https://ieeexplore.ieee.org/document/580378) 的论文中提到，直方图均衡的这种限制的根本原因是没有考虑图片的平均亮度。所以 BHE 就针对这点进行了改进。
> More fundamental reason behind such limitations of the histogram equalization is that the histogram equalization does not take the mean brightness of an image into account.

### Bi-Histogram Equalization(BHE)

用 $X_{m}$ 表示图像 X 的平均值，并假设 $X_{m} ∈\left \{ X_{0}, X_{1}, ..., X_{L-1} \right \}$ ，基于均值可以将图像分为两部分 $X_{L}$ 和 $X_{U}$，整个图片可以表示成：$$X = X_{L}\cup X_{U}$$

然后根据分离的两个子图片，分别求转换方程（cdf），然后再合并 cdf，最后对整个图片进行均衡化。实现代码如下所示：

```python
def BHE(img):
    
    # image mean
    img_mean = int(np.mean(img))
    
    # getting two subimages
    img_l = img.flatten().compress((img.flatten() <= img_mean).flat)
    img_u = img.flatten().compress((img.flatten() > img_mean).flat)
    
    # cdf of low subimage
    hist_l, bins_l = np.histogram(img_l, img_mean+1, [0, img_mean])
    pdf_l = hist_l / np.prod(img_l.size)
    cdf_l = pdf_l.cumsum()

    # transform func of low
    cdf_l = cdf_l *(img_mean - img.min()) + img.min()          
    
    
    # cdf of upper subimage
    hist_u, bins_u = np.histogram(img_u, 256-img_mean, [img_mean+1, 256])
    pdf_u = hist_u / np.prod(img_u.size)
    cdf_u = pdf_u.cumsum()

    # transform func of upper
    cdf_u = cdf_u *(img.max() - (int(img_mean) + 1)) + (int(img_mean) + 1)
    
    cdf_new = np.concatenate((cdf_l, cdf_u))
    new_img = cdf_new[img.ravel()]
    img_eq = np.reshape(new_img, img.shape)
    
    return img_eq
```
用 BHE 得到的结果如下：

![](https://upload-images.jianshu.io/upload_images/2759738-8f120c3b67738221.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-ee4724511be26c84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以发现，BHE 基本保持了原图片的亮度水平，使均衡的结果更自然，而且图片中的一些细节会比原图看的更清楚。

### Clipped Histogram Equalization (CHE)

CHE 是另一种基于 HE 提出的变体算法。基本思想如下图所示：

![](https://upload-images.jianshu.io/upload_images/2759738-e2e0f85b39c55327.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

CHE 会提前指定一个高度，然后大于这个高度的值都会被截取掉，然后将多余的部分均匀地分布在灰度值范围上。

CHE 的缺点很容易发现，需要先 plot 图像的直方图，然后根据直方图的结果手动设置限制的高度范围。同时截取高度再重新分配的操作增加了复杂度。因此，后面提出的 BHEPL 方法会改进这些缺点。

### Bi-Histogram Equalization with a Plateau Limit (BHEPL)

该方法就是在 BHE 的方法中加入了高原限制（上限） $T_{L}$ 和 $T_{U}$：

![](https://upload-images.jianshu.io/upload_images/2759738-5fe85921291c8a38.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-ee4b3f44fc13ea7f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


实际上，$T_{L}$ 是 $h_{L}$ 的平均值， $T_{U}$ 是 $h_{U}$ 的平均值。

下面为了控制增强率，$h_{L}$ 和 $h_{U}$ 会按照下面的公式进行裁剪：

![](https://upload-images.jianshu.io/upload_images/2759738-c84bcb603605bb83.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-f3218f8f1ccd22dd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在切割过程之后，BHEPL 会定义：

![](https://upload-images.jianshu.io/upload_images/2759738-b036101f860226c9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

实际上，$M_{1}$ 是 $X_{L}$ 中所有采样的总数；$M_{2}$ 是 $X_{U}$ 中所有采样的总数。

之后，$X_{L}$ 和 $X_{U}$ 相对的密度函数如下所示：

![](https://upload-images.jianshu.io/upload_images/2759738-bc0790a5a953ea7b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


相对应的累加函数如下所示：

![](https://upload-images.jianshu.io/upload_images/2759738-d0d39cc44469a188.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 结果比较

对于较亮的图像，与 BHE 的结果相比较的话，可以发现，CHE 的结果基本接近 HE 的结果，整体的颜色会偏暗。BHEPL 的结果接近 BHE，保持了原图的亮度，但是结果中会加入一些噪声（也可能是自己代码实现的问题）。

![](https://upload-images.jianshu.io/upload_images/2759738-8a41d75fc1eccac9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于较暗的图像，可以发现 BHEPL 和 CHE 的处理结果都不太好，一个会使整体图片的亮度偏亮，一个会使一些细节部分的亮度变暗。

![](https://upload-images.jianshu.io/upload_images/2759738-f2350389b8b7e321.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面是我在 report 中的结果汇总，分别是亮度值大的图片组，亮度值小的图片组和正常图片组。Adaptive HE 由于没给出论文就没进行总结和实现了，直接使用 OpenCV 库中的函数了。

![](https://upload-images.jianshu.io/upload_images/2759738-7aea933b628b8e13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-a69dacc76c00b14b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/2759738-2b83740b86cdac42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

