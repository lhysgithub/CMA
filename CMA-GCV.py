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
credential_path = "02.json"
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

        # with io.open(file_name, 'rb') as image_file:
        #     content = image_file.read()

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
    QueryTimes = 0
    Topk = 10

    Convergence = 0.01
    # CloseThreshold = - 0.5
    CloseThreshold = 0
    Domin = 0.5
    Sigma = 10
    INumber = 50  # 染色体个数 / 个体个数
    BatchSize = 50  # 寻找可用个体时用的批量上限
    MaxEpoch = 10000  # 迭代上限
    Reserve = 0.25  # 保留率 = 父子保留的精英量 / BestNumber
    BestNmber = int(INumber * Reserve)  # 优秀样本数量
    IndividualShape = (INumber, 299, 299, 3)
    ImageShape = (299, 299, 3)
    StartStdDeviation = 0.1
    CloseEVectorWeight = 0.3
    CloseDVectorWeight = 0.1
    UnVaildExist = 0  # 用来表示是否因为探索广度过大导致无效数据过多
    ConstantUnVaildExist = 0

    # Set output directory
    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)

    # Initialization
    SourceImage = get_image(InputDir,1)
    TargetImage = get_image(InputDir,2)
    SourceType,_ = GCVAPI(SourceImage,OutDir) # 获取首分类
    SourceType = SourceType[0][0] #
    TargetType,_ = GCVAPI(TargetImage,OutDir)
    TargetType = TargetType[0][0]

    # Already done?
    # if (TargetType == SourceType):
    #     print("Done!")

    # 多对多下的初始检验
    for i in SourceType:
        if i in TargetType :
            print("Done!")
            break

    # Set the start point of evolution
    StartImg = StartPoint(SourceImage, TargetImage,Domin)
    Upper = 1.0 - StartImg
    Downer = 0.0 - StartImg


    # Evolution parameters
    SSD = StartStdDeviation
    DM = Domin
    CEV = CloseEVectorWeight
    CDV = CloseDVectorWeight

    StartError = 0 #  unexpectation detection
    LogFile = open(os.path.join(OutDir, 'log%d.txt' % 1), 'w+')
    StartNumber = 2 # the Minimum startnumber  of evolution

    PBF = -1000000.0
    PBL2Distance = 100000
    ENP = np.zeros(ImageShape, dtype=float)
    DNP = np.zeros(ImageShape, dtype=float) + SSD
    # 断点续实验
    if os.path.exists(SourceType[0] + " " + TargetType[0] + "ENP.npy"):
        ENP = np.load(SourceType[0] + " " + TargetType[0] + "ENP.npy")
        DNP = np.load(SourceType[0] + " " + TargetType[0] + "DNP.npy")
    LastENP = ENP
    LastDNP = DNP
    LastPBF = PBF
    LastPBL2 = PBL2Distance

    BestAdv = ENP
    BestAdvL2 = PBL2Distance
    BestAdvF = PBF

    # there is the compute graph
    with tf.Session() as sess:
        Individual = tf.placeholder(shape=IndividualShape, dtype=tf.float32)  # （INumber，299，299，3）
        # logit = tf.placeholder(shape=(INumber), dtype=tf.float32)# We can change the order of response.
        STImg = tf.placeholder(shape=ImageShape, dtype=tf.float32)
        StartImgtf = tf.reshape(STImg,shape= (-1,299,299,3))
        SourceImgtf = tf.placeholder(shape=ImageShape, dtype=tf.float32)
        SourceImg = tf.reshape(SourceImgtf, (-1,299,299,3))
        NewImage = Individual + StartImgtf
        # Labels = tf.constant(1,(0,)*(Topk-1)) # wait to test

        # Compute the L2Distance and IndividualFitness
        L2Distance = tf.sqrt(tf.reduce_sum(tf.square(NewImage - SourceImg), axis=(1, 2, 3)))
        # IndividualFitness = - (Sigma * tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Labels) + L2Distance)
        LossFunction = tf.placeholder(dtype=tf.float32)
        IndividualFitness = - (-LossFunction + L2Distance) # -tf.log(logit)

        # Select BestNmber Individual
        TopKFit, TopKFitIndx = tf.nn.top_k(IndividualFitness, BestNmber)
        TopKIndividual = tf.gather(Individual, TopKFitIndx)  # (BestNmber,299,299,3) 此处是否可以完成

        # Update the Expectation and Deviation
        Expectation = tf.constant(np.zeros(ImageShape), dtype=tf.float32)
        for i in range(BestNmber):
            Expectation += (0.5 ** (i + 1) * TopKIndividual[i])
        # Expectation = tf.reduce_mean(TopKIndividual,reduction_indices=0)
        Deviation = tf.constant(np.zeros(ImageShape), dtype=tf.float32)
        for i in range(BestNmber):
            Deviation += 0.5 ** (i + 1) * tf.square(TopKIndividual[i] - Expectation)
        # Deviation /= BestNmber
        StdDeviation = tf.sqrt(Deviation)

        # Find the best  获取种群最佳（活着的，不算历史的）
        PbestFitness = tf.reduce_max(IndividualFitness)
        Pbestinds = tf.where(tf.equal(PbestFitness, IndividualFitness))
        Pbestinds = Pbestinds[:, 0]
        Pbest = tf.gather(Individual, Pbestinds)

    # Start evolution
    for i in range(MaxEpoch):
        Start = time.time()

        UsefullNumber = 0
        Times = 0
        cycletimes = 0
        initI = np.zeros(IndividualShape, dtype=float)
        # initCp = np.zeros((INumber), dtype=float)
        initCR = np.zeros((INumber), dtype=float)
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

            # CP = np.reshape(CP, (BatchSize, 1000))
            
            # 筛选
            QueryTimes += BatchSize
            for j in range(BatchSize):
                if TargetType in PP[j]:
                    initI[UsefullNumber] = TempPerturbation[j]
                    # templabes = [-1]*len(PP[j])
                    # templabes[PP[j].index(TargetType)] = 10
                    # templabes[PP[j].index(SourceType)] = -10
                    # initLoss[UsefullNumber] = -np.sum(np.log(CP[j])*templabes)
                    initLoss[UsefullNumber] = np.sum(np.log(CP[j][PP[j].index(TargetType)])-np.log(CP[j]))
                    # initLoss[UsefullNumber] = np.log(np.exp(CP[j][PP[j].index(TargetType)])/np.sum(np.exp(np.log(CP[j]))))
                    # initLoss[UsefullNumber] = np.exp(CP[j][PP[j].index(TargetType)])/np.sum(np.exp(np.log(CP[j])))
                    initCR[UsefullNumber] = np.log(CP[j][PP[j].index(TargetType)]/CP[j][0])
                    # initCp[UsefullNumber] = CP[j][PP[j].index(TargetType)]
                    UsefullNumber += 1
                    if UsefullNumber == INumber:
                        break

            # Check whether the UsefullNumber equals INumber
            if UsefullNumber != INumber:
                LogText = "UsefullNumber: %3d SSD: %.2f DM: %.3f" % (UsefullNumber, SSD, DM)
                LogFile.write(LogText + '\n')
                print(LogText)

            # if we find some usefull individual we can find more
            if UsefullNumber > StartNumber - 1 and UsefullNumber < INumber:
                tempI = initI[0:UsefullNumber]
                ENP = np.zeros(ImageShape, dtype=float)
                DNP = np.zeros(ImageShape, dtype=float)
                for j in range(UsefullNumber):
                    ENP += tempI[j]
                ENP /= UsefullNumber
                for j in range(UsefullNumber):
                    DNP += np.square(tempI[j] - ENP)
                DNP /= UsefullNumber
                DNP = np.sqrt(DNP)

            # We need to find some init-usefull individual
            if i == 0 and UsefullNumber < StartNumber:
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

            # If invalid happened, we need to roll back dnp and enp
            if i != 0 and UsefullNumber < StartNumber:
                CEV -= 0.01
                CDV = CEV / 3
                if CEV <= 0.01:
                    CEV = 0.01
                    CDV = CEV / 3
                DNP = LastDNP + (SourceImage - (StartImg + ENP)) * CDV
                ENP = LastENP + (SourceImage - (StartImg + ENP)) * CEV
                LogText = "UnValidExist CEV: %.3f CDV: %.3f" % (CEV, CDV)
                LogFile.write(LogText + '\n')
                print(LogText)

            # 判断是否出现样本无效化
            if cycletimes == 0:
                if i != 0 and UsefullNumber < StartNumber:
                    UnVaildExist = 1
                elif i != 0 and UsefullNumber >= StartNumber:
                    UnVaildExist = 0
            cycletimes += 1

            # Check whether the ssd overflows
            if SSD > 1:
                LogText = "Start Error"
                LogFile.write(LogText + '\n')
                print(LogText)
                StartError = 1
                break

        # Error dispose
        if StartError == 1:
            break

        # initI = np.clip(initI, Downer, Upper)

        LastPBF, LastDNP, LastENP = PBF, DNP, ENP
        PBI,ENP, DNP, PBF, PB = sess.run([Pbestinds,Expectation, StdDeviation, PbestFitness, Pbest],
                                             feed_dict={Individual: initI, LossFunction: initLoss,STImg:StartImg,SourceImgtf:SourceImage})

        # 断点续实验
        np.save(SourceType[0] + " " + TargetType[0] + "ENP.npy", ENP)
        np.save(SourceType[0] + " " + TargetType[0] + "DNP.npy", DNP)

        PBI = PBI[0]
        if PB.shape[0] > 1:
            PB = PB[0]
            PB = np.reshape(PB, (1, 299, 299, 3))
            print("PBConvergence")

        End = time.time()
        LastPBL2 = PBL2Distance
        PBL2Distance = np.sqrt(np.sum(np.square(StartImg + PB - SourceImage), axis=(1, 2, 3)))

        render_frame(OutDir, PB, 100 + i, SourceType, TargetType, StartImg)
        LogText = "Step %05d: PBF: %.4f UseingTime: %.4f PBL2Distance: %.4f QueryTimes: %d" % (
            i, PBF, End - Start, PBL2Distance, QueryTimes)
        LogFile.write(LogText + '\n')
        print(LogText)

        # elif i>10 and LastPBF > PBF: # 发生抖动陷入局部最优(不应该以是否发生抖动来判断参数，而是应该以是否发现出现无效数据来判断，或者两者共同判断)
        if PBL2Distance>15 and abs(PBF - LastPBF) < Convergence:
            if (PBF + PBL2Distance > CloseThreshold):  # 靠近
            # if ( 1 ):  # 靠近
                CEV += 0.01
                CDV = CEV / 3
                DNP += (SourceImage - (StartImg + ENP)) * CDV
                ENP += (SourceImage - (StartImg + ENP)) * CEV
                LogText = "Close up CEV: %.3f CDV: %.3f" % (CEV, CDV)
                LogFile.write(LogText + '\n')
                print(LogText)
            else:
                # CEV += 0.01
                # CDV = CEV / 3
                DNP += (SourceImage - (StartImg + ENP)) * CDV
                LogText = "Scaling up CEV: %.3f CDV: %.3f" % (CEV, CDV)
                LogFile.write(LogText + '\n')
                print(LogText)

        if (initPP[PBI][0] not in TargetType) and PBL2Distance < 15 and abs(PBF - LastPBF) < Convergence:
            # CEV += 0.01
            # CDV = CEV / 3
            DNP += (SourceImage - (StartImg + ENP)) * CDV
            LogText = "Scaling up CEV: %.3f CDV: %.3f" % (CEV, CDV)
            LogFile.write(LogText + '\n')
            print(LogText)

        # 如果结果还行，可以保存
        # if (PBF + PBL2Distance > CloseThreshold):  # 靠近
        # if initCR[PBI]>=0:
        if initPP[PBI][0] in TargetType:
            BestAdv = PB
            BestAdvL2 = PBL2Distance
            BestAdvF = PBF

        # 解雇
        if BestAdvL2 < 15:
            LogText = "Complete BestAdvL2: %.4f BestAdvF: %.4f QueryTimes: %d" % (
                BestAdvL2, BestAdvF, QueryTimes)
            print(LogText)
            LogFile.write(LogText + '\n')
            render_frame(OutDir, BestAdv, 1000000, SourceType, TargetType, StartImg)
            break

        # 最大循环次数
        if i == MaxEpoch - 1 or ConstantUnVaildExist == 30:
            LogText = "Complete to MaxEpoch or ConstantUnVaildExist BestAdvL2: %.4f BestAdvF: %.4f QueryTimes: %d" % (
                BestAdvL2, BestAdvF, QueryTimes)
            print(LogText)
            LogFile.write(LogText + '\n')
            render_frame(OutDir, BestAdv, 1000000, SourceType, TargetType, StartImg)
            break

if __name__ == '__main__':
    main()
