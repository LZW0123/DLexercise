import tensorflow.compat.v1 as tf #type:ignore
import numpy as np
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

def AddLayer(input,inputShape,outputShape,activation_function=None):
    w=tf.Variable(tf.random.normal([inputShape,outputShape]))
    b=tf.Variable(tf.zeros([1,outputShape])+0.1)
    y=tf.matmul(input,w)+b
    if activation_function is None:
        return y
    else:
        return activation_function(y)

# 创建输入数据
xData=np.linspace(0,1,100)[:,np.newaxis] # 从shape(100,)变为(100,1)
# print(xData.shape)
noise=np.random.normal(0,1,(100,1))
yData=np.square(xData)+1+noise

# 定义输入数据 占位符
xs=tf.placeholder(tf.float32,shape=[None,1])
ys=tf.placeholder(tf.float32,shape=[None,1])

                            ############ 创建层 ###########

hiddenLayer=AddLayer(xs,1,10,activation_function=tf.nn.relu) # 定义一个隐含层
outputLayer=AddLayer(hiddenLayer,10,1,activation_function=None) # 定义一个输出层

                            ########## 求解神经网络参数 #########
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-outputLayer),reduction_indices=[0])) # 定义损失函数
trainStep=tf.train.GradientDescentOptimizer(0.01).minimize(loss) # 定义训练过程
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

                                ####### 进行训练 #########
for i in range(10000):
    sess.run(trainStep,feed_dict={xs:xData,ys:yData})
    if i%100==0:
        print(sess.run(loss,feed_dict={xs:xData,ys:yData}))

                                ######## 关闭sess ########
sess.close()

