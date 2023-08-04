from tensorflow.keras import Input, Sequential, Model, layers
from DataSetMaker.Layer import stockLoRA

def makeModel(
        total_used_stock,
        dim_latent = 16,
        dim_embedM = 4,
        dim_embedW = 4,
        waveNetDilation = 2,
        waveNetDepth = 3,
        rankS = 1,
):
    inputMonth = Input(shape=(None,),name='month_index')
    inputWeekday = Input(shape=(None,),name='weekday_index')
    embededMonth = layers.Embedding(12,dim_embedM,name='month_embedding')(inputMonth)
    embededWeekday = layers.Embedding(5,dim_embedW,name='weekday_embedding')(inputWeekday)

    inputEt = Input(shape=(None,window_len),name='E_t')
    inputBetaRest = Input(shape=(None,degree),name='Beta_rest')

    patternEncoder = Sequential([
        layers.Dense(2*dim_latent,activation='relu',name='pattern_encode_2'),
        layers.Dense(dim_latent,activation='relu',use_bias=False,name='pattern_encode_3'),
        layers.LayerNormalization(name='pattern_normalize'),
    ],name='pattern_encoder')

    trendEncoder = Sequential([
        layers.Dense(2*dim_latent,activation='relu',name='trend_encode_1'),
        layers.Dense(dim_latent,activation='relu',use_bias=False,name='trend_encode_2'),
        layers.LayerNormalization(name='trend_normalize'),
    ],name='trend_encoder')

    patternLatent = patternEncoder(inputEt)
    trendLatent = trendEncoder(inputBetaRest)
    series_latent = layers.Add(name='series_latent')([patternLatent,trendLatent])

    latent_transfer = Sequential([
        layers.Dense(2*dim_latent,activation='relu',name='latent_1',),
        layers.Dense(dim_latent,activation='relu',use_bias=False,name='latent_2'),
        layers.LayerNormalization(name='latent_normalize'),
    ],name='latent_transfer')

    latent_concat = layers.Concatenate(name='latent_concat')
    latent_all = latent_concat([embededMonth,embededWeekday,patternLatent,trendLatent])
    latent_pre = layers.Add(name='latent')([series_latent,latent_transfer(latent_all)])

    latent_waveNet = Sequential([
        layers.Conv1D(
            waveNetDilation**(waveNetDepth-1-i)*dim_latent,
            waveNetDilation,dilation_rate=waveNetDilation,padding='causal',
            activation='relu',name=f'wave_net_{i+1}',)
        for i in range(waveNetDepth)
    ],name='wave_net')

    latent = layers.Add(name='after_waveNat')([latent_pre,latent_waveNet(latent_pre),series_latent])

    trendDecoder = Sequential([
        layers.Dense(dim_latent,activation='relu',name='trend_decode_1'),
        layers.Dense(dim_latent,activation='relu',name='trend_decode_2'),
    ],name='trand_decoder')

    inputStock = Input(shape=(),name='stock_index')
    LoRAStock = stockLoRA(total_used_stock,dim_latent,rankS,name='stock_LoRA')
    adapted_latent = layers.Add(name='after_LoRA')([latent,trendDecoder(latent),LoRAStock(latent,inputStock),series_latent])

    outputTrend = layers.Dense(3,name='score')(adapted_latent)

    MyModel = Model((inputEt,inputBetaRest,
                     inputMonth,inputWeekday,inputStock),
                   outputTrend,
                   name='model_for_trend')
    return MyModel
    
