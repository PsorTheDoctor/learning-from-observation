import tensorflow as tf
from models.model import AlexNet, TestModel
from models.vit import DeepViT
# from models.swin import SwinTransformerModel
import pandas as pd
import numpy as np
import datetime
from const import *
import os

# https://www.inference.org.uk/itprnn/book.pdf
JOINT_PATH = "data/snake/joints.npy"
IMAGES_PATH = "data/snake/images.npy"
VALIDATION_SIZE = 100
BATCH_SIZE = 64

def vit():

    # read images and joint positions
    joint_pos = np.reshape(np.load(JOINT_PATH), (-1, 2))
    images = np.load(IMAGES_PATH)

    validation_data = (images[0:VALIDATION_SIZE], joint_pos[0:VALIDATION_SIZE])
    # remove validation data from training data
    images = images[VALIDATION_SIZE:]
    joint_pos = joint_pos[VALIDATION_SIZE:]

    # create model vit
    model = DeepViT(
        image_size = 224,
        patch_size = 28,
        num_classes = 2,
        dim = 1024, # 26x26
        depth = 3,
        heads = 10,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    # create loss function
    loss_fn = tf.keras.losses.Huber()

    # create tensorboard
    log_dir = "logs/vit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)
    writer.set_as_default()

    #set default tensorboard
    tf.summary.trace_on(graph=True, profiler=True)

    # custom training loop
    for i in range(10000):
        # random BATCH_SIZE indexes for batch processing
        indexes = np.random.randint(0, len(images), BATCH_SIZE)
        # get batch images and joint positions
        batch_images = images[indexes]
        batch_joint_pos = joint_pos[indexes]
        with tf.GradientTape() as tape:
            out = model(batch_images)
            loss = loss_fn(batch_joint_pos, out)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # tensorboard
        tf.summary.scalar('loss', loss, step=i)
        tf.summary.histogram('out', out, step=i)
        tf.summary.histogram('joint_pos', batch_joint_pos, step=i)
        # validation step
        if i % 10 == 0:
            # random BATCH_SIZE indexes for batch processing
            indexes = np.random.randint(0, VALIDATION_SIZE, BATCH_SIZE)
            # get batch images and joint positions
            batch_images = validation_data[0][indexes]
            batch_joint_pos = validation_data[1][indexes]
            out = model(batch_images)
            loss = loss_fn(batch_joint_pos, out)
            tf.summary.scalar('val_loss', loss, step=i)
            tf.summary.histogram('val_out', out, step=i)
            tf.summary.histogram('val_joint_pos', batch_joint_pos, step=i)

def swin():
    #read images and joint positions
    joint_pos = np.reshape(np.load(JOINT_PATH), (-1, 2))
    images = np.load(IMAGES_PATH)

    validation_data = (images[0:VALIDATION_SIZE], joint_pos[0:VALIDATION_SIZE])
    #remove validation data from training data
    images = images[VALIDATION_SIZE:]
    joint_pos = joint_pos[VALIDATION_SIZE:]

    #create model swin
    model = SwinTransformerModel(
        image_size = (224,224),
        patch_size = (28,28),
        num_classes=2,
        include_top=True,
        window_size=8,
        ape = True,
        embed_dim = 192)

    #create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    #create loss function
    loss_fn = tf.keras.losses.Huber()

    #create tensorboard
    log_dir = "logs/swin/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)
    writer.set_as_default()

    #set default tensorboard
    tf.summary.trace_on(graph=True, profiler=True)

    #custom training loop
    for i in range(10000):
        #random BATCH_SIZE indexes for batch processing
        indexes = np.random.randint(0, len(images), BATCH_SIZE)
        #get batch images and joint positions
        batch_images = images[indexes]
        batch_joint_pos = joint_pos[indexes]
        with tf.GradientTape() as tape:
            out = model(batch_images)
            loss = loss_fn(batch_joint_pos, out)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #tensorboard
        tf.summary.scalar('loss', loss, step=i)
        tf.summary.histogram('out', out, step=i)
        tf.summary.histogram('joint_pos', batch_joint_pos, step=i)
        #validation step
        if i % 10 == 0:
            #random BATCH_SIZE indexes for batch processing
            indexes = np.random.randint(0, VALIDATION_SIZE, BATCH_SIZE)
            #get batch images and joint positions
            batch_images = validation_data[0][indexes]
            batch_joint_pos = validation_data[1][indexes]
            out = model(batch_images)
            loss = loss_fn(batch_joint_pos, out)
            tf.summary.scalar('val_loss', loss, step=i)
            tf.summary.histogram('val_out', out, step=i)
            tf.summary.histogram('val_joint_pos', batch_joint_pos, step=i)

if __name__ == "__main__":
    vit()