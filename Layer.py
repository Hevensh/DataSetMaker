import tensorflow as tf
from tensorflow.keras import Model, layers

class StockLoRA(Model):
    def __init__(
        self,
        num_stocks,
        dim_latent,
        rank=2,
        **kwargs,):
        super().__init__(**kwargs)
        self.num_stocks = num_stocks
        self.embedA = layers.Embedding(num_stocks,dim_latent*rank,)
        self.embedB = layers.Embedding(num_stocks,dim_latent*rank,
                                       embeddings_initializer='zeros',)
        self.beta = layers.Embedding(num_stocks,dim_latent,
                                       embeddings_initializer='zeros',)
        self.reshape = layers.Reshape((dim_latent,rank))
        
    def call(self, latent, indexStock):
        loraA = self.reshape(self.embedA(indexStock))
        loraB = self.reshape(self.embedB(indexStock))
        beta = self.beta(indexStock)
        r = tf.einsum('bij,bjk->bik',latent,loraA)
        outputs = tf.einsum('bij,bkj->bik',r,loraB)
        return outputs + beta[:,None]


class DateEmbbeding(Model):
    def __init__(
        self,
        num_embeds,
        dims,
        **kwargs,):
        super().__init__(**kwargs)
        self.num_embeds = len(num_embeds)
        self.embeds = [layers.Embedding(i, dims) for i in num_embeds]
        self.sum = layers.Add()
        
    def call(self, inputs):
        outputs = [
            embed_i(input_i)
            for input_i, embed_i 
            in zip(tf.unstack(inputs, axis=-1), self.embeds)
        ]

        return self.sum(outputs)
