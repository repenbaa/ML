import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow.keras as keras
# %run reader.ipynb

import winsound


def Beep():
    frequency = 2000
    duration = 1000
    winsound.Beep(frequency, duration)


Beep()

class_dim = 5
EPOCHS = 100
BATCH_SIZE = 32
init_model = None

model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(128, None, 1)),
    tf.keras.layers.ActivityRegularization(l2=0.5),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
])

model.summary()


# 定义优化方法
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 讀取處理後的訓練資料
b = './audio_sets/'

train_dataset = train_reader_tfrecord(
    b + 'train.tfrecord', EPOCHS, batch_size=BATCH_SIZE)
test_dataset = test_reader_tfrecord(b + 'test.tfrecord', batch_size=BATCH_SIZE)

if init_model:
    model.load_weights(init_model)

for batch_id, data in enumerate(train_dataset):
    # [可能需要修改参数】 设置的梅尔频谱的shape
    sounds = data['data'].numpy().reshape((-1, 128, 128, 1))
    labels = data['label']
    # 执行训练
    with tf.GradientTape() as tape:
        predictions = model(sounds)
        # 获取损失值
        train_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions)
        train_loss = tf.reduce_mean(train_loss)
        # 获取准确率
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
            labels, predictions)
        train_accuracy = np.sum(train_accuracy.numpy()) / \
            len(train_accuracy.numpy())

    # 更新梯度
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 進度
    if batch_id % 20 == 0:
        print("Batch %d, Loss %f, Accuracy %f" %
              (batch_id, train_loss.numpy(), train_accuracy))

    # 保存模型
    if batch_id % 200 == 0 and batch_id != 0:
        test_losses = list()
        test_accuracies = list()
        for d in test_dataset:
            # [可能需要修改参数】 设置的梅尔频谱的shape
            test_sounds = d['data'].numpy().reshape((-1, 128, 128, 1))
            test_labels = d['label']

            test_result = model(test_sounds)
            # 获取损失值
            test_loss = tf.keras.losses.sparse_categorical_crossentropy(
                test_labels, test_result)
            test_loss = tf.reduce_mean(test_loss)
            test_losses.append(test_loss)
            # 获取准确率
            test_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
                test_labels, test_result)
            test_accuracy = np.sum(test_accuracy.numpy()) / \
                len(test_accuracy.numpy())
            test_accuracies.append(test_accuracy)

        print('=================================================')
        print("Test, Loss %f, Accuracy %f" % (
            sum(test_losses) / len(test_losses), sum(test_accuracies) / len(test_accuracies)))
        print('=================================================')

        model.save('./models/model.h5')
        model.save_weights('./models/model_weights.h5')

model.save('./models/model.h5')
model.save_weights('./models/model_weights.h5')

Beep()

# 從tf中拿x y train test
#
train_dataset = train_reader_tfrecord(
    '../DataSets/OurVoice/train.tfrecord', 1, batch_size=1)
test_dataset = test_reader_tfrecord(
    '../DataSets/OurVoice/test.tfrecord', batch_size=1)

X_train = []
X_test = []
for batch_id, data in tqdm(enumerate(train_dataset)):
    # [可能需要修改参数】 设置的梅尔频谱的shape
    sounds = data['data'].numpy()
    labels = data['label']
    X_train.append(sounds)
    X_test.append(labels)

y_train = []
y_test = []
for batch_id, data in tqdm(enumerate(test_dataset)):
    # [可能需要修改参数】 设置的梅尔频谱的shape
    sounds = data['data']
    labels = data['label']
    y_train.append(sounds)
    y_test.append(labels)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.shape, X_test.shape

model.predict(X_train, X_test)
