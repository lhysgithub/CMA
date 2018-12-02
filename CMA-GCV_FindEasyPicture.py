import numpy as np
import tensorflow as tf
import os
import io
import time # Used to count using time
import PIL # import image
import shutil # DELETE DIRECTORY
import scipy # draw render
import matplotlib.pyplot as plt # draw render
from google.cloud import vision # GCV
from google.cloud.vision import types # GCV

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
credential_path = "04.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# parallel computeing
def GCVAPI(image,OutDir):

    image = np.reshape(image,(-1,299,299,3))
    labels = []
    confidences =[]
    print(len(image))

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    for i in range(len(image)):
        # Save the image as .jpg
        scipy.misc.imsave(os.path.join(OutDir, '%s.jpg' % i), image[i])

        # The name of the image file to annotate
        file_name = os.path.join(os.path.dirname(__file__), os.path.join(OutDir, '%s.jpg' % i))

        # parallel query

        # Read the image file
        with open(file_name, 'rb') as image_file:
            content = image_file.read()
        # image_file.closed

        # Binary to Image
        testimage = types.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=testimage)
        templabs = response.label_annotations
        tempDescriptions = []
        tempScores = []
        for j in templabs:
            tempDescriptions.append(j.description)
            tempScores.append(j.score)
        labels.append(tempDescriptions)
        confidences.append(tempScores)

    return labels,confidences
    # (INumber,Topk),(INumber,Topk)返回一个字典类型

def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width) / 2)
        image = image.crop((0, height_off, image.width, height_off + image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height) / 2)
        image = image.crop((width_off, 0, width_off + image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:, :, :3]
    return img

def get_image(InputDir="", indextemp=-1):
    image_paths = sorted([os.path.join(InputDir, i) for i in os.listdir(InputDir)])

    if indextemp != -1:
        index = indextemp
    else:
        index = np.random.randint(len(image_paths))

    path = image_paths[index]
    x = load_image(path)
    return x


def render_frame(OutDir, image, save_index, SourceClass, TargetClass, StartImg):
    image = np.reshape(image, (299, 299, 3)) + StartImg
    scipy.misc.imsave(os.path.join(OutDir, '%s.jpg' % save_index), image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    # image
    ax1.imshow(image)
    fig.sca(ax1)
    plt.xticks([])
    plt.yticks([])

    # classifications
    probs,confidence = GCVAPI(image,OutDir)
    probs = probs[0]
    confidence= confidence[0]

    barlist = ax2.bar(range(len(probs)), confidence)
    # 多对多的染色方案
    for i, v in enumerate(probs):
        if v in SourceClass:
            barlist[i].set_color('g')
        elif v in TargetClass:
            barlist[i].set_color('r')

    # 一对一时的染色方案
    # for i, v in enumerate(probs):
    #     if v == SourceClass:
    #         barlist[i].set_color('g')
    #     if v == TargetClass:
    #         barlist[i].set_color('r')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(len(probs)), probs, rotation='vertical')
    fig.subplots_adjust(bottom=0.2)

    path = os.path.join(OutDir, 'frame%06d.jpg' % save_index)
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()


def StartPoint(SourceImage, TargetImage, Domin):
    StartUpper = np.clip(TargetImage + Domin, 0.0, 1.0)
    StartDowner = np.clip(TargetImage - Domin, 0.0, 1.0)
    ClipedImage = np.clip(SourceImage, StartDowner, StartUpper)
    return ClipedImage

def main():

    # Algorithm parameters
    InputDir = "adv_samples/"
    OutDir = "adv_example/"
    StartDir = "adv_start/"
    QueryTimes = 0
    # Set output directory
    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)

    # 遍历所有的源目图片组
    for i in range(100):
        sourceIndex = int(i/10)
        targetIndex = i%10

        # 如果原图等于目标图片，跳过
        if sourceIndex == targetIndex:
            continue

        # 一些必须的实验参数
        Continue = 1 # 断点续实验开关
        Domin = 0.5
        INumber = 50  # 染色体个数 / 个体个数
        BatchSize = 50  # 寻找可用个体时用的批量上限
        IndividualShape = (INumber, 299, 299, 3)
        ImageShape = (299, 299, 3)
        StartStdDeviation = 0.1

        # Initialization
        SourceImage = get_image(InputDir,sourceIndex)
        TargetImage = get_image(InputDir,targetIndex)

        # 确定两张输入图片的识别分类
        SourceType, _ = GCVAPI(SourceImage, OutDir)  # 获取首分类
        TargetType, _ = GCVAPI(TargetImage, OutDir)

        # 手动选择原始分类
        # print("SourceType:")
        # for keyword in range(len(SourceType[0])):
        #     print("%d. "%keyword + SourceType[0][keyword])
        # SourceIndex = input("Please input the index of source type:")

        # 自动选择原始分类
        SourceIndex = 0

        # 手动选择目标分类
        # print("TargetType:")
        # for keyword in range(len(TargetType[0])):
        #     print("%d. "%keyword + TargetType[0][keyword])
        # TargetIndex = input("Please input the index of target type:")

        # 自动选择目标分类
        TargetIndex = 0

        TempSourceType=[]
        TempTargetType= []
        TempSourceType.append(SourceType[0][int(SourceIndex)])   #
        TempTargetType.append(TargetType[0][int(TargetIndex)])
        SourceType = TempSourceType
        TargetType = TempTargetType

        # Set the start point of evolution
        # 断点续实验
        if Continue==1 and os.path.exists(StartDir+SourceType[0] + " " + TargetType[0] + ".jpg"):
            StartImg = get_image(StartDir,0)
        elif Continue == 1 and os.path.exists(StartDir+SourceType[0] + " " + TargetType[0] + "Start.npy"):
            StartImg = np.load(StartDir+SourceType[0] + " " + TargetType[0] + "Start.npy")
        else:
            StartImg = StartPoint(SourceImage, TargetImage,Domin)
        Upper = 1.0 - StartImg
        Downer = 0.0 - StartImg

        # Evolution parameters
        SSD = StartStdDeviation
        DM = Domin

        LogFile = open(os.path.join(OutDir, 'log%d.txt' % i), 'w+')
        StartNumber = 2 # the Minimum startnumber  of evolution

        ENP = np.zeros(ImageShape, dtype=float)
        # 断点续实验
        if Continue==1 and os.path.exists(StartDir+SourceType[0] + " " + TargetType[0] + "DNP.npy"):
            DNP = np.load(SourceType[0] + " " + TargetType[0] + "DNP.npy")
        else:
            DNP = np.zeros(ImageShape, dtype=float) + SSD

        UsefullNumber = 0
        Times = 0
        initI = np.zeros(IndividualShape, dtype=float)
        initPP = []
        initLoss = np.zeros((INumber), dtype=float)

        # find the usefull Individual
        while UsefullNumber != INumber:

            # Generate TempPerturbation, TestImage, CP and PP
            TempPerturbation = np.random.randn(BatchSize, 299, 299, 3)
            TempPerturbation = TempPerturbation * np.reshape(DNP, (1, 299, 299, 3)) + np.reshape(ENP, (1, 299, 299, 3))
            TempPerturbation = np.clip(TempPerturbation, Downer, Upper)
            TestImage = TempPerturbation + np.reshape(StartImg, (1, 299, 299, 3))
            PP, CP = GCVAPI(TestImage,OutDir)
            Used = np.zeros(BatchSize)

            # 筛选
            QueryTimes += BatchSize
            for j in range(BatchSize):
                for oneTType in TargetType:
                    if (oneTType in PP[j]) and Used[j]==0:

                        initI[UsefullNumber] = TempPerturbation[j]
                        initPP.append(PP[j])

                        templabes = [-1] * len(PP[j])
                        for k in PP[j]:
                            if k in TargetType:
                                templabes[PP[j].index(k)] = 10
                            elif k in SourceType:
                                templabes[PP[j].index(k)] = -10

                        # 对数正数的交叉熵
                        initLoss[UsefullNumber] = np.sum((np.log(CP[j]))*templabes)
                        # 对数倒数的交叉熵
                        # initLoss[UsefullNumber] = - np.sum((1 / np.log(CP[j]))*templabes)

                        Used[j]=1
                        UsefullNumber += 1
                        if UsefullNumber == INumber: # 找够了，跳出有效进化
                            break
                if UsefullNumber == INumber: # 找够了，跳出有效进化
                    break

            # Check whether the UsefullNumber equals INumber
            if UsefullNumber != INumber:
                LogText = "Number %d UsefullNumber: %3d SSD: %.2f DM: %.3f" % (i,UsefullNumber, SSD, DM)
                LogFile.write(LogText + '\n')
                print(LogText)

            # We need to find some init-usefull individual
            if UsefullNumber < StartNumber:
                Times += 1
                TimesUper = 1
                if UsefullNumber > 0:
                    TimesUper = 5
                else:
                    TimesUper = 1

                if Times == TimesUper:
                    SSD += 0.01
                    if SSD - StartStdDeviation >= 0.05:
                        SSD = StartStdDeviation
                        DM -= 0.05
                        StartImg = StartPoint(SourceImage, TargetImage,DM)
                        Upper = 1.0 - StartImg
                        Downer = 0.0 - StartImg

                    DNP = np.zeros(ImageShape, dtype=float) + SSD
                    Times = 0

            # Check whether the ssd overflows
            if SSD > 1:
                LogText = "Start Error"
                LogFile.write(LogText + '\n')
                print(LogText)
                StartError = 1
                break

            # 判断是否找到初始种群
            if UsefullNumber >= StartNumber:
                break
        LogFile.close()
        

if __name__ == '__main__':
    main()
