from __future__ import absolute_import, print_function
from logging import PlaceHolder

import tvm
import numpy as np
from tvm import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.te.create_schedule(C.op)
    # xo, xi = s[C].split(s[C].op.axis[0], factor=32)
    # s[C].reorder(xi, xo)
    # s[C].bind(xo, tvm.te.thread_axis("blockIdx.x"))
    # s[C].bind(xi, tvm.te.thread_axis("threadIdx.x"))
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(A.shape, lambda *i: A(*i) + const_k)
    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A , B], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(A.shape, lambda *i: A(*i) * const_k)
    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A , B], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(A.shape, lambda *i: tvm.te.max(tvm.runtime.const(0, A.dtype),A(*i)),name="B")
    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A , B], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(shape,lambda *i:B(*i) * tvm.tir.Select(A(*i)>0 , tvm.runtime.const(1, A.dtype) , tvm.runtime.const(0, A.dtype)) , name="C")
    s = tvm.te.create_schedule(C.op)
    
    f = tvm.build(s, [A , B , C], tgt, target_host=tgt_host, name=func_name)
    return f
    
def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.te.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.te.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.te.placeholder(shapeB, dtype=dtype, name="B")
    if transposeA is False and transposeB is False:
        # print(shapeA,shapeB)
        k = tvm.te.reduce_axis((0,shapeA[1]),name='k')
        C = tvm.te.compute((shapeA[0],shapeB[1]), lambda x,y: tvm.te.sum(A[x,k]*B[k,y],axis=k),name='C')
    elif transposeA is False and transposeB is True:
        k = tvm.te.reduce_axis((0,shapeA[1]),name='k')
        C = tvm.te.compute((shapeA[0],shapeB[0]), lambda x,y: tvm.te.sum(A[x,k]*B[y,k],axis=k),name='C')
    elif transposeA is True and transposeB is False:
        k = tvm.te.reduce_axis((0,shapeA[0]),name='k')
        C = tvm.te.compute((shapeA[1],shapeB[1]), lambda x,y: tvm.te.sum(A[k,x]*B[k,y],axis=k),name='C')
    elif transposeA is True and transposeB is True:
        k = tvm.te.reduce_axis((0,shapeA[0]),name='k')
        C = tvm.te.compute((shapeA[1],shapeB[0]), lambda x,y: tvm.te.sum(A[k,x]*B[y,k],axis=k),name='C')
    s = tvm.te.create_schedule(C.op)

###################################
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
    ko, ki = s[C].split(k, factor=8)
    s[C].reorder(xo, yo, ko, xi, yi, ki)
    s[C].parallel(xo)
    s[C].unroll(ki)
###################################
    # bn=32
    # xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    # k, = s[C].op.reduce_axis
    # ko, ki = s[C].split(k, factor=4)

    # # re-ordering,改变xi和ki的循环位置
    # s[C].reorder(xo, yo, ko, xi, ki, yi)
    # s[C].vectorize(yi)
##############################
    # bn = 32
    # #通过循环平铺tile来分块
    # xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)#32x32的块，返回内外循环索引
    # k, = s[C].op.reduce_axis
    # ko, ki = s[C].split(k, factor=4)
    # #ko,ki移到内外块循环之间
    # s[C].reorder(xo, yo, ko, ki, xi, yi)
###############################    
    # (x, y), (k,) = C.op.axis, C.op.reduce_axis
    # s[C].reorder(x, k, y)
##################################
    # xo, xi = s[C].split(C.op.axis[1], factor=16)
    # s[C].reorder(k, xi, xo)
    # s[C].vectorize(xi)
    # s[C].parallel(xo)
    # print(tvm.lower(s, [A, B , C], simple_mode=True))
    f = tvm.build(s,[A , B , C],tgt,target_host=tgt_host,name=func_name)
    return f 

def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.te.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    X = tvm.te.placeholder(shapeX, dtype=dtype, name="A")
    F = tvm.te.placeholder(shapeF, dtype=dtype, name="B")

    channel = tvm.te.reduce_axis((0, C), "channel")
    di = tvm.te.reduce_axis((0, R), "di")
    dj = tvm.te.reduce_axis((0, S), "dj")
    shapeOut = (shapeX[0],shapeF[0],H-R+1,W-S+1)
    Y = tvm.te.compute(shapeOut,lambda n,m,i,j:tvm.te.sum(X[n,channel,i+di,j+dj]*F[m,channel,di,dj],axis=[channel,di,dj]),name='Y')

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s,[X , F , Y],tgt,target_host=tgt_host,name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.te.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    k1 = tvm.te.reduce_axis((0,shape[1]),name="k1")
    k2 = tvm.te.reduce_axis((0,shape[1]),name="k2")
    maxA = tvm.te.compute((shape[0],),lambda i:tvm.te.max(A[i,k1],axis=k1),name="maxA")
    eA = tvm.te.compute(shape,lambda i,j:tvm.te.exp(A[i,j]-maxA[i]),name="eA")
    sumA = tvm.te.compute((shape[0],),lambda i:tvm.te.sum(eA[i,k2],axis=k2),name="sumA")
    softA = tvm.te.compute(shape,lambda i,j:eA[i,j]/(sumA[i]),name="softA")
    
    s = tvm.te.create_schedule(softA.op)
    f = tvm.build(s,[A , softA],tgt,target_host=tgt_host,name=func_name)
    return f

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")

    k1 = tvm.te.reduce_axis((0,shape[1]),name="k1")
    k2 = tvm.te.reduce_axis((0,shape[1]),name="k2")
    maxA = tvm.te.compute((shape[0],),lambda i:tvm.te.max(A[i,k1],axis=k1),name="maxA")
    eA = tvm.te.compute(shape,lambda i,j:tvm.te.exp(A[i,j]-maxA[i]),name="eA")
    sumA = tvm.te.compute((shape[0],),lambda i:tvm.te.sum(eA[i,k2],axis=k2),name="sumA")
    logA = tvm.te.compute(shape,lambda i,j:A[i,j]-maxA[i]-tvm.te.log(sumA[i]),name="logA")
    
    k3 = tvm.te.reduce_axis((0,shape[1]),name="k3")
    k4 = tvm.te.reduce_axis((0,shape[0]),name="k4")
    crossA = tvm.te.compute((shape[0],),lambda i:tvm.te.sum(B[i,k3]*logA[i,k3],axis=k3),name="crossA")
    ans = tvm.te.compute((1,),lambda i:tvm.te.sum(crossA[k4],axis=k4),name="ans")
    output = tvm.te.compute((1,),lambda i:-ans[i]/shape[0],name='output')

    s = tvm.te.create_schedule(output.op)
    f = tvm.build(s,[A , B , output],tgt,target_host=tgt_host,name=func_name)
    return f 


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.te.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.te.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f