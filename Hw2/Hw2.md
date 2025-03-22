# 																	Hw2

## 任务描述

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250321113206880.png" alt="image-20250321113206880" style="zoom:67%;" />



> ![image-20250321113443876](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250321113443876.png)
>
> 把一段语音，拿到它的频谱之后，取一定的时间间隔（25ms），把它分成许多的小块。 对于切的每一小块，经过了滤波操作，最后经过DCT的变换，就拿到了MFCC向量。

> **重点！！！**
>
> ![image-20250322104100527](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250322104100527.png)
>
> 向神经网络传入数据的时候，并不是将这些所谓的帧（向量）挨个排队进入，而是先进行flatten（压平）为一个长的向量，进入。  
>
> 取相邻帧可以提高预测的精准度。

> 
>
> <img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250321114133578.png" alt="image-20250321114133578" style="zoom:50%;" />
>
> 原始语音，每25ms切一部分，切成对应的一个帧，再把窗口向后挪10ms，再切一个，拿到第二个帧，按照此过程依次往后挪。
>
> 最后，我们就把这段语音给切割成了**由T个39维的MFCD向量构成的语音文件**。

> ![image-20250321113805345](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250321113805345.png)
>
> 我们将n个39维的向量传入神经网络，然后预测属于什么类别。

> ![image-20250321233120246](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250321233120246.png)

>
>
>![image-20250322161852367](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250322161852367.png)
>
>网络模型结构



> ![image-20250322163946172](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250322163946172.png)



