import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
import datetime
import json
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange


class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)
    
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.fn.save_weights(filepath, overwrite, save_format, options)
    
    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.fn.load_weights(filepath, by_name)


class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return nn.Activation(gelu)

        self.net = [
            nn.Dense(units=hidden_dim),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)
    
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.net.save_weights(filepath, overwrite, save_format, options)
    
    def load_weights(self, filepath, by_name=False):
        self.net.load_weights(filepath, by_name)


class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.reattn_weights = tf.Variable(initial_value=tf.random.normal([heads, heads]))

        self.reattn_norm = [
            Rearrange('b h i j -> b i j h'),
            nn.LayerNormalization(),
            Rearrange('b i j h -> b h i j')
        ]

        self.to_out = [
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]

        self.reattn_norm = Sequential(self.reattn_norm)
        self.to_out = Sequential(self.to_out)
    
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        temp_value = self.reattn_weights.value()
        with open(filepath + '_reattn_weights', 'w') as f:
            json.dump(temp_value.numpy().tolist(), f)

        self.to_out.save_weights(filepath + '_to_out_', overwrite, save_format, options)
    
    def load_weights(self, filepath, by_name=False):
        loaded_value = None
        with open(filepath + '_reattn_weights', 'r') as f:
            loaded_value = json.load(f)

        self.reattn_weights.assign(loaded_value)
        self.to_out.load_weights(filepath + '_to_out_', by_name)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # attention
        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.attend(dots)

        # re-attention
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out
        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)
        return x


class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x

        return x
    
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # create dir if not exists
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, layer in enumerate(self.layers):
            attn, mlp = layer
            attn.save_weights(filepath + '/attn_{}.h5'.format(i), overwrite=overwrite, save_format=save_format, options=options)
            mlp.save_weights(filepath + '/mlp_{}.h5'.format(i), overwrite=overwrite, save_format=save_format, options=options)
    
    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        for i, layer in enumerate(self.layers):
            attn, mlp = layer
            attn.load_weights(filepath + '/attn_{}.h5'.format(i))
            mlp.load_weights(filepath + '/mlp_{}.h5'.format(i))


class DeepViT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):
        super(DeepViT, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Dense(units=dim)
        ], name='patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='mlp_head')

        # save parameters to local variables
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.dropout_param = dropout
        self.emb_dropout = emb_dropout 

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)
        return x
    
    def save(self, path):
        self.patch_embedding.save_weights(path + '/patch_embedding')
        self.mlp_head.save_weights(path + '/mlp_head')
        self.transformer.save_weights(path + '/transformer')
        np.save(path + '/cls_token.npy', self.cls_token.numpy())
        np.save(path + '/pos_embedding.npy', self.pos_embedding.numpy())
        # save parameters
        with open(path + '/parameters.json', 'w') as f:
            json.dump({
                'image_size': self.image_size,
                'patch_size': self.patch_size,
                'num_patches': self.num_patches,
                'dim': self.dim,
                'heads': self.heads,
                'dim_head': self.dim_head,
                'mlp_dim': self.mlp_dim,
                'pool': self.pool,
                'dropout_param': self.dropout_param,
                'emb_dropout': self.emb_dropout
            }, f)
    
    def load(self, path):
        # first call for initializing variables
        self(tf.zeros([1, self.image_size, self.image_size, 3]))
        #load parameters
        with open(path + '/parameters.json', 'r') as f:
            parameters = json.load(f)

        self.dropout = nn.Dropout(rate=parameters['emb_dropout'])
        self.patch_embedding.load_weights(path + '/patch_embedding')
        self.mlp_head.load_weights(path + '/mlp_head')
        self.transformer.load_weights(path + '/transformer')
        self.cls_token = tf.Variable(initial_value=np.load(path + '/cls_token.npy'))
        self.pos_embedding = tf.Variable(initial_value=np.load(path + '/pos_embedding.npy'))
    
    def fit(self, epochs, joints_path, images_path, validation_size=100, batch_size=64, learning_rate=5e-5,
            loss_fn=tf.keras.losses.Huber(), log_dir="logs/vit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            save_path='model/vit'):

        # read images and joint positions
        joint_pos = np.reshape(np.load(joints_path), (-1, 2))
        images = np.load(images_path)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # create tensorboard
        writer = tf.summary.create_file_writer(log_dir)
        writer.set_as_default()

        #set default tensorboard
        tf.summary.trace_on(graph=True, profiler=True)

        # custom training loop
        for i in range(epochs):
            # random BATCH_SIZE indexes for batch processing
            indexes = np.random.randint(validation_size, len(images), batch_size)

            # get batch images and joint positions
            batch_images = images[indexes]
            batch_joint_pos = joint_pos[indexes]
            with tf.GradientTape() as tape:
                out = self(batch_images)
                loss = loss_fn(batch_joint_pos, out)

            grads = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # tensorboard
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.histogram('out', out, step=i)
            tf.summary.histogram('joint_pos', batch_joint_pos, step=i)

            # validation step
            if i % 10 == 0:
                # random BATCH_SIZE indexes for batch processing
                indexes = np.random.randint(0, validation_size, batch_size)

                # get batch images and joint positions
                batch_images = images[indexes]
                batch_joint_pos = joint_pos[indexes]
                out = self(batch_images)
                loss = loss_fn(batch_joint_pos, out)
                tf.summary.scalar('val_loss', loss, step=i)
                tf.summary.histogram('val_out', out, step=i)
                tf.summary.histogram('val_joint_pos', batch_joint_pos, step=i)
        
        self.save(save_path)


""" Usage
v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
img = tf.random.normal(shape=[1, 256, 256, 3])
preds = v(img) # (1, 1000)
"""