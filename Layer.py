import tensorflow as tf
from tensorflow.keras import Model, layers

class StockLoRA(Model):
    def __init__(
        self,
        num_stocks,
        dim_latent,
        rank=2,
        **kwargs,):
        super().__init__()
        self.num_stocks = num_stocks
        self.embedA = layers.Embedding(num_stocks,dim_latent*rank,)
        self.embedB = layers.Embedding(num_stocks,dim_latent*rank,
                                       embeddings_initializer='zeros',)
        self.reshape = layers.Reshape((dim_latent,rank))
        
    def call(self, latent, indexStock):
        loraA = self.reshape(self.embedA(indexStock))
        loraB = self.reshape(self.embedB(indexStock))
        r = tf.einsum('bij,bjk->bik',latent,loraA)
        outputs = tf.einsum('bij,bkj->bik',r,loraB)
        return outputs
      
