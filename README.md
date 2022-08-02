# CurricularFace-Loss

arcFace = adding_penalty_of_ground_true_at_cosine_similarity

CurricularFace = arcFace + adding_weight_of_hard_sample

---

Under pictures are embeddings about mnist dataset after training. 

### Original Softmax Loss
![download](https://user-images.githubusercontent.com/48349693/182266855-eb57ee1d-f140-4500-af7b-fa1d5870160e.png)

### CurricularFace Loss
![download](https://user-images.githubusercontent.com/48349693/182266736-44cb7dc4-b273-45cc-9ff6-e3fd7672b0a5.png)

---

# Code Explanation

### Load mnist dataset
```python
import tensorflow as tf
import numpy as np

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
print(train_x.shape, train_y.shape)
print(train_x.dtype, train_y.dtype)

train_x = train_x.astype(np.float32) / 255.0
test_x = test_x.astype(np.float32) / 255.0

train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)
print(train_y.shape, train_y.dtype)
print(test_y.shape, test_y.dtype)
```
Load mnist dataset.

### CurricularFace Loss
```python
class CurricularFaceLoss(tf.keras.losses.Loss):
    def __init__(self, scale=30, margin=0.5, alpha=0.99, name="CurricularFaceLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.t = tf.Variable(0.)
        self.eps = 1e-7
    
    def positive_forward(self, y_logit):
        cosine_sim = y_logit 
        theta_margin = tf.math.acos(cosine_sim) + self.margin
        y_logit_pos = tf.math.cos(theta_margin)
        return y_logit_pos
    
    def negative_forward(self, y_logit_pos_masked, y_logit):
        hard_sample_mask = y_logit_pos_masked < y_logit # (N, n_classes)
        y_logit_neg = tf.where(hard_sample_mask, tf.square(y_logit)+self.t*y_logit, y_logit)
        return y_logit_neg
    
    def forward(self, y_true, y_logit):
        y_logit = tf.clip_by_value(y_logit, -1.0+self.eps, 1.0-self.eps)
        y_logit_masked = tf.expand_dims(tf.reduce_sum(y_true*y_logit, axis=1), axis=1) # (N, 1)
        y_logit_pos_masked = self.positive_forward(y_logit_masked) # (N, 1)
        y_logit_neg = self.negative_forward(y_logit_pos_masked, y_logit) # (N, n_classes)
        # update t
        r = tf.reduce_mean(y_logit_pos_masked)
        self.t.assign(self.alpha*r + (1-self.alpha)*self.t)
        
        y_true = tf.cast(y_true, dtype=tf.bool)
        return tf.where(y_true, y_logit_pos_masked, y_logit_neg)
    
    def __call__(self, y_true, y_logit): # shape(N, n_classes)
        y_logit_fixed = self.forward(y_true, y_logit)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_logit_fixed*self.scale)
        loss = tf.reduce_mean(loss)
        return loss
```
positive_forward = cos(Θ + m)

negative_forward = cos(Θ)² + t×cos(Θ) if HARD_SAMPLE else cos(Θ)


### Cosine Similarity Layer
```python
class CosSimLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, regularizer=None, name='NormLayer', **kwargs):
        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._regularizer = regularizer

    def build(self, embedding_shape):
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')

    def call(self, embedding, training=None):
        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits') # (N, dim)
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights') # (dim, 10)
        cosine_sim = tf.matmul(x, w, name='cosine_similarity') # (N, 10)
        return cosine_sim, x
```
I returned normalized_embedding because of drawing plot.

##### inner product formula
```
A · B = |A|×|B|×cos(Θ) = a1×b1 + a2×b2 + ... + an×bn
if |A| == |B| == 1:
    cos(Θ) = a1×b1 + a2×b2 + ... + an×bn = A · B
```


---

### Reference

https://emkademy.medium.com/angular-margin-losses-for-representative-embeddings-training-arcface-2018-vs-mv-arc-softmax-96b54bcd030b

https://linuxtut.com/en/a9dadc68a5cd0a2747c0/




