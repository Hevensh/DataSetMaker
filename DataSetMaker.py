import numpy as np

class DataSetMaker:
    def __init__(
            self,
            figure_series,
            date_series,
            window_len=32,
    ):
        self.date_series = date_series
        self.window_len = window_len
        ns2day = 24 * 60 * 60 * 1e9

        self.seriesX = figure_series
        self.seriesDate = np.double(date_series) / ns2day

        self.embedDate = self.date_series[window_len - 1:]

        self.MakeSlices = MakeSlices
        self.Detrend = Detrend
        self.IndexDate = IndexDate
        self.TakeALook = TakeALook
