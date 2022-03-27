import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers
from tensorflow import keras

# 加载数据集
(xs,ys),_=datasets.mnist.load_data()

print('datasets:',xs.shape,ys.shape) # 打印出数据的维度信息

# 装数据转化为tensor类型，处理更高效
xs=tf.convert_to_tensor(xs,dtype=tf.float32)/255. # 并归一化到0~1
ys=tf.convert_to_tensor(ys,dtype=tf.int32)
ys=tf.one_hot(ys,depth=10)


# 转化为tensorflow的数据集形式，对数据进行batch，同时计算多个样本
db=tf.data.Dataset.from_tensor_slices((xs,ys)).batch(23) # batch()方法对多个样本分为一组进行训练，

# 784->512->256->10 这样一个降维过程
# for step,(x,y) in enumerate(db):
#     print(step,x.shape,y,y.shape)

model=keras.Sequential([                   # 创建每一层的节点数
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='relu')
])
# 选择优化器
optimizer=optimizers.SGD(learning_rate=0.001)


def train_model(epoch):
    for step, (x, y) in enumerate(db):
        with tf.GradientTape() as tape:
            # 将[b,28,28] 变成 [b,784]
            x = tf.reshape(x, (-1, 28 * 28))
            # step1,计算输出，[b,784]->[b,10]
            out = model(x)
            # step2,计算loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        # step3,计算梯度
        grads = tape.gradient(loss, model.trainable_variables)
        # step4，更新参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step%100==0:
            print(epoch,step,loss.numpy())

def train():
    for epoch in range(30):
        train_model(epoch)

if __name__=='__main__':
    train()