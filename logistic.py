import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils_logistic import load_dataset

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

m_train = train_set_y.shape[1]       #训练集里图片的数量。
m_test = test_set_y.shape[1]         #测试集里图片的数量。
num_px = train_set_x_orig.shape[1]   #训练、测试集里面的图片的宽度和高度（均为64x64）。

#将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


#求解sigmord函数
def sigmord(z):
    a=1/(1+np.exp(-z))
    return a
#求z
#def function_z(W,X,b):
#    Z=np.dot(W.T,X)+b
#    return Z

#初始化w，b
def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=float(0)
    return w,b

def train(X,Y,W,b):
    m=X.shape[1]
    Z=np.dot(W.T,X)+b
    A=sigmord(Z)
    dz=A-train_set_y
    dw=(1/m)*np.dot(X,dz.T)
    db=1/m*np.sum(dz)
    J=-1/m*np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))
    return dw,db,J

def logistic(X,Y,W,b,times,learn_rate):
    for i in range(1,times+1):
        dw,db,lost=train(X,Y,W,b)
        W-=dw*learn_rate
        b-=db*learn_rate
        if i%100==0:
            print("第%i次训练后的准确率为%f"%(i,1-lost))
    return W,b

def predict(X,W,b):
    m = X.shape[1]
    Y_predict=np.zeros((1,m))
    Z = np.dot(W.T, X) + b
    A = sigmord(Z)
    for i in range(m):
        Y_predict[0][i]=1 if A[0][i]>0.7 else 0
    return Y_predict

def main():
    Ww,Bb=initialize_with_zeros(train_set_x.shape[0])
    times=eval(input("请输入迭代次数"))
    learn_rate=eval(input("请输入学习率"))
    W,b=logistic(train_set_x,train_set_y ,Ww,Bb,times,learn_rate)
    print("\n训练完成\n")
    Y_predict_train=predict(train_set_x,W,b)
    Y_predict_text =predict(test_set_x ,W, b)
    print("训练集准确率：{:.4f}%".format((1-np.mean(np.abs(Y_predict_train-train_set_y)))*100))
    print("测试集准确率：{:.4f}%".format((1-np.mean(np.abs(Y_predict_text -test_set_y))) * 100))

main()