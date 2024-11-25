from svmMLiA import *
from numpy import *

def simpleSMO(dataMatIn, classLabels, C, tolerant, maxIter):
    """
    简化后的SMO算法
    :param dataMatIn:   数据矩阵
    :param classLabels: 类别标签
    :param C:           松弛变量
    :param tolerant:    容错率
    :param maxIter:     最大迭代次数
    """
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMat)
    alphas = mat(zeros(m, 1))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMat *dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if (labelMat(labelMat[i] * Ei < -tolerant) and (alphas[i] < C)) or \
               ((labelMat[i] * Ei > tolerant) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print(f"L==H")
                    continue
                eta = 2.0 * dataMat[i,:] * dataMat[j,:].T - dataMat[i,:] * dataMat[i,:].T - dataMat[j,:] * dataMat[j,:].T
                if eta >= 0:
                    print(f"eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print(f"j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i,:]*\
                    dataMat[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMat[i,:] * dataMat[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i,:] * dataMat[j,:].T -\
                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[j,:] * dataMat[j,:].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print(f"iter: %d i:%d, pairs changed %d"%(iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print(f"iteration number: %d"%(iter))
    return b, alphas
                
