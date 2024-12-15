from svmMLiA import *
from numpy import *

def simpleSMO(dataMatIn, classLabels, C, tolerant, maxIter):
    """
    Function: 简化后的SMO算法
    input:    dataMatIn:   数据矩阵
              classLabels: 类别标签
              C:           松弛变量
              tolerant:    容错率
              maxIter:     最大迭代次数
    output:   b:           常数项
              alphas:      数据向量
    """
    # 将数据集转换为numpy矩阵
    dataMat = mat(dataMatIn)
    # 将类别标签转换为numpy矩阵并转制为列向量
    labelMat = mat(classLabels).transpose()
    # 初始化常数项b为0
    b = 0
    # 获得数据集中样本的数量和特征的数量
    m, n = shape(dataMat)
    # 初始化数据向量为m维0向量
    alphas = mat(zeros((m,1)))
    # 初始化迭代次数为0
    iter = 0
    while iter < maxIter:
        # alpha优化标志位在每次循环开始时都置0
        alphaPairsChanged = 0
        # 内层循环
        for i in range(m):
            # 将alphas和labelMat相乘，求出法向量w(m, 1)，w`(1, m)
            # dataMatr * dataMat[i,:].T，求出输入向量x(m, 1)
            # 整个计算对应公式f(x) = w`x + b，求取的是间隔
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            # 计算误差
            Ei = fXi - float(labelMat[i])
            # 对符合约束条件的样本进行优化：如果标签和误差相乘的结果在容忍范围之外，且对应的alpha值在常数范围内，则进行优化
            if ((labelMat[i] * Ei < -tolerant) and (alphas[i] < C)) or \
               ((labelMat[i] * Ei > tolerant) and (alphas[i] > 0)):
                # 随机选择另一个向量
                j = selectJrand(i, m)
                # 对该向量重复间隔和误差的计算
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 使用copy存储就的alpha值便于后续比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证优化后的alpha值在0-C之间
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果界限值相同，则立即开启下一次循环
                if L == H:
                    print(f"L==H")
                    continue
                # 最优修改量，计算两个向量的内积(核函数)
                eta = 2.0 * dataMat[i,:] * dataMat[j,:].T - dataMat[i,:] * dataMat[i,:].T - dataMat[j,:] * dataMat[j,:].T
                # 如果最优修改量大于0，则不进行优化，跳过本次循环：这里是简化处理了Platt的SMO算法
                if eta >= 0:
                    print(f"eta>=0")
                    continue
                # 计算新的alpha[j]值，并作限幅处理
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果新旧alpha值变化很小，则不进行优化，跳过本次循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print(f"j not moving enough")
                    continue
                # 计算新的alpha[i]值，计算量相同，优化方向与alpha[j]相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 计算新的常数项b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i,:]*\
                    dataMat[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMat[i,:] * dataMat[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i,:] * dataMat[j,:].T -\
                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[j,:] * dataMat[j,:].T
                # 选择符合约束条件的alpha值对应的常数项作为最终的常数项，如果两个alpha值都超出了约束范围就取平均值
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # alpha对优化次数+1
                alphaPairsChanged += 1
                print(f"iter: %d i:%d, pairs changed %d"%(iter, i, alphaPairsChanged))
        # 如果没有对alpha值优化，则迭代次数+1；否则置0
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print(f"iteration number: %d"%(iter))
    return b, alphas
                
def testSMO():
    dataArr, labelArr = loadDataSet('/home/pazfic/Git_ws/Machine_Learning/SVM/testSet.txt')
    b, alphas = simpleSMO(dataArr, labelArr, 0.6, 0.001, 40)
    

if __name__ == '__main__':
    testSMO()
