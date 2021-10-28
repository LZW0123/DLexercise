# 加载库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,optimizers

# 加载数据
(xs,ys),_=datasets.mnist.load_data() # 还必须加上_

# 转换数据至tensor类型
xs=tf.convert_to_tensor(xs,dtype=tf.float32)/255.
ys=tf.convert_to_tensor(ys,dtype=tf.int32)
ys=tf.one_hot(ys,depth=10) # 编码为one_hot编码
db=tf.data.Dataset.from_tensor_slices((xs,ys)).batch(23) # 将数据集进行划分

# 选择优化器
optimizer=optimizers.SGD(learning_rate=0.001)

# 创建Model
model=keras.Sequential(
    [layers.Dense(512,activation='relu'),
     layers.Dense(256,activation='relu'),
     layers.Dense(10,activation='relu')]
)
# 定义训练模型
def train_model(epoch):
    for step,(x,y) in enumerate(db):
        with tf.GradientTape() as tape:
            x=tf.reshape(x,(-1,28*28))
            out=model(x)
            loss=tf.reduce_sum(tf.square(out-y))/x.shape[0]
        grad=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        if step%100==0:
            print(epoch,step,loss.numpy())

def train():
    for epoch in range(20):
        train_model(epoch)

if __name__=='__main__':
    train()