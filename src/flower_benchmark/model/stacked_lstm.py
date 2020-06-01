# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stacked LSTM"""
from typing import Optional, Tuple

import tensorflow as tf
#from tensorflow.contrib import rnn

def orig_lstm(
    input_len, hidden_size: int, num_classes: int,  seed: int
) -> tf.keras.Model:
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_len)  # input_len = 80
    layers = tf.keras.layers.LSTM(units=hidden_size)(inputs)
    layers = tf.keras.layers.LSTM(units=hidden_size)(layers)
    outputs = tf.keras.layers.Dense(
        num_classes, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model w/ learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


""" Model from Leaf

def create_model(self):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8]) # whats 8 for?
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32) # _ = state, tensor of shape [batch_size, cell_state_size]
        pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss

shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
seq_len = 80 = max length sentence we want fo r asingle input in characters
outputs' is a tensor of shape [batch_size, max_time, cell_state_size
"""
