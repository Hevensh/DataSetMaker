import numpy as np
import matplotlib.pyplot as plt

class DataSetMaker:
    def __init__(
        self,
        figure_series,
        date_series,
    ):
        self.date_series = date_series
        ns2day = 24 * 60 * 60 * 1e9

        self.seriesX = figure_series
        self.seriesDate = np.double(date_series) / ns2day        
    
    def MakeSlices(
        self,
        window_len=32,
    ):
        self.window_len = window_len
        self.embedDate = self.date_series[window_len - 1:]
        
        data_len = self.seriesX.shape[0]
    
        sample_len = data_len - window_len + 1
        self.sample_len = sample_len
    
        slicesX = np.zeros([sample_len, window_len])
        slicesDate = np.zeros([sample_len, window_len])
    
        for i in range(window_len):
            slicesX[:, i] = self.seriesX[i:i + sample_len]
            slicesDate[:, i] = self.seriesDate[i:i + sample_len]
        slicesDate -= slicesDate[:, -1:]
    
        self.slicesX = slicesX
        self.slicesDate = slicesDate
        return slicesX, slicesDate


    def Detrend(
        self,
        useRealGap=True,
        useWeights=True,
        temperature=77,  # higher the tempe, lower the effect of weight
        degree=2,
    ):
        self.useRealGap = useRealGap
        self.useWeights = useWeights
    
        self.temperature = temperature
        self.degree = degree

        poly_degree = degree + 1
        sample_len = self.sample_len
        window_len = self.window_len
    
        if useRealGap:
            base_pattern = self.slicesDate / window_len
    
            window_pattern = np.zeros([sample_len, poly_degree, window_len])
            for i in range(poly_degree):
                window_pattern[:, i, :] = base_pattern ** i
    
        else:
            base_pattern = np.arange(1 - window_len, 1) / window_len
    
            window_pattern = np.zeros([poly_degree, window_len])
            for i in range(poly_degree):
                window_pattern[i] = base_pattern ** i
    
        if useWeights:
            weight = temperature / (temperature - base_pattern * window_len)
        else:
            weight = np.eye(window_len)
    
        if useRealGap:
            TrendDatas = np.zeros([sample_len, poly_degree])
            TrendOfSlicesX = np.zeros_like(self.slicesX)
    
            for i in range(sample_len):
                tempWeight = np.diag(weight[i])
                moment_estimate_matrix = np.dot(
                    window_pattern[i].T,
                    np.linalg.inv(window_pattern[i] @ tempWeight @ window_pattern[i].T)
                )

                TrendDatas[i] = slicesX[i] @ tempWeight @ moment_estimate_matrix
                TrendOfSlicesX[i] = TrendDatas[i] @ window_pattern[i]
    
        else:
            tempWeight = np.diag(weight)
            moment_estimate_matrix = np.dot(
                window_pattern.T,
                np.linalg.inv(window_pattern @ tempWeight @ window_pattern.T)
            )
    
            TrendDatas = slicesX @ tempWeight @ moment_estimate_matrix
    
            TrendOfSlicesX = TrendDatas @ window_pattern
        deTrendedX = slicesX - TrendOfSlicesX
    
        self.base_pattern = base_pattern
        self.TrendDatas = TrendDatas
        self.TrendOfSlicesX = TrendOfSlicesX
        return deTrendedX, TrendDatas

    
    def IndexDate(
        self,
    ):
        indexMonth = (self.embedDate.dt.month-1).to_numpy()
        indexWeekday = (self.embedDate.dt.weekday).to_numpy()
        return indexMonth, indexWeekday

    
    def TakeALook(
        self,
        i = 0,
    ):
        if self.useRealGap:
            tempPattern = self.base_pattern[i]
        else:
            tempPattern = self.base_pattern
    
        plt.plot(tempPattern,self.slicesX[i],'b-',
            tempPattern,self.TrendOfSlicesX[i],'g--',)
        plt.legend(['real','trend'])
        plt.title(f'''the {i}-th slice ,
    real gap: {self.useRealGap} ,
    weighted: {self.useWeights} .''')
    plt.show()
        
