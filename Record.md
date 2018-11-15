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

```python
initLoss[UsefullNumber] = - np.sum((1 / np.log(CP[j]))*templabes)
initLoss < - 10 # 触发“收敛”操作
Convergence = 0.01
CloseEVectorWeight = 0.3
CloseDVectorWeight = 0.1
CloseThreshold = 10
- initLoss > 10 
if (PBF + PBL2Distance> CloseThreshold):  # 靠近
```
${1/{\log{0.95}}=-44.90}$

${1/{\log{0.90}}=-21.85}$

${1/{\log{0.85}}=-14.17}$

${1/{\log{0.80}}=-10.32}$

实验效果：

​	成功完成了目标分类排名上升，原始分类排名下降，但是目前还存在一定的问题。

问题及分析：

- 目标分类依排名依然不能成为第一。我们可以看到在本实验结果中，目标分类与原始分类的置信度已经及其相近，我相信随着算法的推演，目标分类成为第一是有可能的。我们可能忽略一个重要的事实，那就是对于不同的原始图片与目标图片，对抗样本生成难度是不一样的。因此，我们有必要挑选一组攻击较为容易的原始图片与目标图片组。
- 由于“收敛”的条件太苛刻了，所以算法一直不能触发“靠近”操作。我们可以通过已有实验数据判断，如何设置Convergence和CloseThreshold比较合理。

解决方案：

​	我们直观的选择一个较好的可以触发“靠近”操作的样本，再具体分析其Convergence和CloseThreshold特点。

##### lab18

```python
initLoss[UsefullNumber] = - np.sum((1 / np.log(CP[j]))*templabes)
CloseEVectorWeight = 0.3
CloseDVectorWeight = 0.1
if (PBF + PBL2Distance> CloseThreshold):  # 靠近

initLoss < 70 # 触发“收敛”操作
Convergence = 2
CloseThreshold =  - 70
- initLoss >  - 70
SourceImage = get_image(InputDir,7)
TargetImage = get_image(InputDir,6)
```

实验内容：

- 测试从“装甲车”向“卡丁车”的攻击组。
- 测试新的“收敛”触发条件是否具有好的效果。





