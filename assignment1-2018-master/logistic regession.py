import autodiff as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(x):
    w = np.random.random([x])
    b = np.random.random([1])
    return w,b
def normalize(x):
    diff = np.max(x)-np.min(x)
    x = (x-np.min(x))/diff
    return x

def read_data(path):
    data = pd.read_csv(path,header=None)
    x = data.iloc[:,0:-1].to_numpy()
    y= data.iloc[:,-1].to_numpy()
    x[:,0]=normalize(x[:,0])
    x[:,1]=normalize(x[:,1])
    return x,y

def accuracy(w_in,x_in,b_in,y_in):
    y = 1 / (np.exp(-(x_in @ w_in + b_in)) + 1)
    y = (y >= 0.5) == y_in
    return np.sum(y) / len(y_in)

def show_result(w_in, x_in, b_in, y_in):
    x1 = np.linspace(np.min(x_in[:, 0]), np.max(x_in[:, 0]), 100)
    x2 = -(w_in[0] * x1 + b_in) / w_in[1] 
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(y_in)):
        if y_in[i] == 1:
            ax.scatter(x_in[i][0], x_in[i][1], s=50, c='b', marker='o')
        else:
            ax.scatter(x_in[i][0], x_in[i][1], s=50, c='r', marker='x')
    ax.plot(x1, x2, label='Decision Boundary', c='grey')
    ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()

def show_loss(loss_list):
    plt.plot(np.arange(0, len(loss_list)), loss_list)
    plt.ylabel('loss')
    plt.show()

def show_acc(acc_list):
    plt.plot(np.arange(0, len(acc_list)), acc_list)
    plt.ylabel('acc')
    plt.show()
def train(n=100, lr=0.01 , epochs=200):
    w = ad.Variable(name = 'w')
    x = ad.Variable(name = 'x')
    b = ad.Variable(name = 'b')
    y = ad.Variable(name = 'y')

    o = ad.reduce_sum_op(w * x) + b
    h = 1 / (ad.exp_op(-o) + 1 )
    l = -(y*ad.ln_op(h)+(1+(-y))*ad.ln_op(1+(-h)))

    grad_w , grad_b = ad.gradients(l,[w,b])

    
    x_in,y_in = read_data('test.txt')
    w_in,b_in = generate_data(x_in.shape[1])

    executor=ad.Executor([l,grad_w,grad_b])

    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        l_sum = 0
        for i in range(n):
            l_val,grad_w_val,grad_b_val = executor.run(feed_dict={w:w_in,x:x_in[i],b:b_in,y:y_in[i]})
            w_in -= lr * grad_w_val
            b_in -= lr * grad_b_val
            l_sum+=l_val
        loss_list.append(l_sum)
        acc = accuracy(w_in,x_in,b_in,y_in)
        acc_list.append(acc)
        print("epoch: {0}, loss: {1}, acc: {2}".format(epoch,l_sum,acc))

    show_loss(loss_list)
    show_acc(acc_list)
    show_result(w_in, x_in, b_in, y_in)
if __name__ ==  '__main__' :
    train()
