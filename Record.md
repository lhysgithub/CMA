##### lab5

- Add the sigma = 10 to improve the effection of classfication.
- Let the lab of source-type become -2
- Let the ending condition more toughly. Only the BestAdvL2 < 15, we done.

##### lab6

- change the lab of source-type to -10 and change the lab of target-type to 10

已经能进入前五了，但是还不能到第一，或许，只有第一次到了第一才能比较好的进化。

##### lab16

```python
initLoss[UsefullNumber] = np.sum(np.log(CP[j][PP[j].index(TargetType)])-np.log(CP[j]))
Convergence = 0.01
CloseThreshold = 0
CloseEVectorWeight = 0.3
CloseDVectorWeight = 0.1
# The CEV and CDV arn't increasing when Scaling up 
# CEV += 0.01
# CDV = CEV / 3
```

实验内容：

​	由于定向靠近后目标分类排名难以提高到第一名，我们采取了措施。我们使得”靠近“的触发条件更加苛刻，即唯有目标分类排名到达了第一名之后才能触发”靠近“操作。

​	为了配合上述苛刻条件，我们将收敛条件也设置的更为苛刻。即，唯有前后两轮适应度之差小于0.01时，判定为CMA进化已“收敛”。

实验效果：

- “靠近”操作后陷入局部最优。靠近操作后很难再触发靠近操作。
- 原始分类的分类排名基本未变。
- 80轮后，目标分类依旧不能进入前5，并且原始分类的分类排名也基本未变

实验分析：

​	这种损失函数设置方式不能很好的提升目标分类置信度，降低目标分类置信度。因此我们必须改回之前的损失函数设置方式。

##### lab17