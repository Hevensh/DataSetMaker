import time
import numpy as np
import pandas as pd
import tensorflow as tf

from IPython import display
from DataSetMaker.DataSetMaker import DataSetMaker

def poltRateOfProcess(k, total, time, process='processing', filename='file', several=False):
    display.clear_output(wait=True)
    if several:
        the_string = [None] * len(k)
        for i in range(len(k)):
            c = k[i] + 1
            current = 20 * c // total[i]
            if current < 20:
                the_string[i] = (process[i] + ' [' + '=' * current +
                                 '>' + '-' * (19 - current) + ']' +
                                 f' - {c}/{total[i]}  ' +
                                 f'{int(time[i])}s {int(time[i] * 1e3 / c)}ms/' +
                                 filename[i])
            else:
                the_string[i] = (process[i] + ' is done [' + '=' * 20 + ']' +
                                 f' - {c}/{total[i]}  ' +
                                 f'{int(time[i])}s {int(time[i] * 1e3 / c)}ms/' +
                                 filename[i])

        print_string = the_string[0]
        for i in range(1, len(k)):
            print_string += '\n' + the_string[i]
        print(print_string)

    else:
        c = k + 1
        current = 20 * c // total
        if current < 20:
            print(
                process + ' [' + '=' * current +
                '>' + '-' * (19 - current) + ']' +
                f' - {c}/{total}  ' +
                f'{int(time)}s {int(time * 1e3 / c)}ms/' +
                filename
            )
        else:
            print(
                process + ' is done [' + '=' * 20 + ']' +
                f' - {c}/{total}  ' +
                f'{int(time)}s {int(time * 1e3 / c)}ms/' +
                filename
            )


def ilocSeries(data0):
    ns2day = 24 * 60 * 60 * 1e9
    dates = pd.to_datetime(data0['Date'], utc=True)
    dates_numpy = np.double(dates) / ns2day

    series = np.log(data0['Close'])
    return dates_numpy, series, dates


class DataLoader:
    def __init__(
            self,
            window_len: int = 32,  # window length for segment
            pred_days: int = 16,  # predict length for segment
    ):
        self.window_len = window_len
        self.pred_days = pred_days

    def loadTheData(
            self,
            file_list,
            last_day='2018-1-1',  # the train data is before the last day
        
            pre_process_fun=ilocSeries,
            # fun need to return:
            # seriesDate (numpy type)
            # series_list (numpy type)
            # date_list
    ):
        ns2day = 24 * 60 * 60 * 1e9
        self.train_last_day = np.double(pd.Timestamp(last_day).to_numpy()) / ns2day

        self.total_num_stock = len(file_list)
        self.series_list = [None] * self.total_num_stock
        self.date_list = [None] * self.total_num_stock
        self.train_length = np.zeros(self.total_num_stock, np.int32)
        self.val_length = np.zeros(self.total_num_stock, np.int32)

        start = time.time()
        for chosen in range(self.total_num_stock):
            data0 = pd.read_csv(file_list[chosen])

            self.seriesDate, self.series_list[chosen], self.date_list[chosen] = pre_process_fun(data0)

            self.train_length[chosen] = (self.seriesDate < self.train_last_day).sum()
            self.val_length[chosen] = (self.seriesDate.shape[0] -
                                       self.train_length[chosen] - self.pred_days)

            end = time.time()
            poltRateOfProcess(chosen, self.total_num_stock, end - start, 'loading')

    def chooseFeasibleData(
            self,
            min_num_segment=1,  # the min num of segments can be sliced in one stock
            length_segment=2000,  # the length of each segment
    ):
        self.min_num_segment = min_num_segment
        self.length_segment = length_segment

        self.feasible_index = self.train_length > (length_segment * min_num_segment + self.window_len)

        self.feasible_train_len = self.train_length[self.feasible_index] - self.window_len + 1
        self.feasible_val_len = self.val_length[self.feasible_index]

        self.num_segment_each_stock = self.feasible_train_len // length_segment
        self.total_used_stock = self.num_segment_each_stock.shape[0]
        self.total_segments = self.num_segment_each_stock.sum()

        self.valMin = np.unique(self.feasible_val_len).min() - self.pred_days

        print(f'{self.total_segments} segments from {self.total_used_stock} feasible stocks,')
        print(f'each segment has {self.length_segment} samples pairs.')
        print(f'validation for each stock has {self.valMin} samples pairs.')

        self.feasible_series = [None] * self.total_used_stock
        self.feasible_date = [None] * self.total_used_stock

        pos = 0
        for i in range(len(self.series_list)):
            if self.feasible_index[i]:
                self.feasible_series[pos] = self.series_list[i]
                self.feasible_date[pos] = self.date_list[i]
                pos += 1

    def processTheData(
            self,
            use_real_gap=True,  # if use real date gap when detrend
            use_weights=True,  # if use weights when detrend
            temperature=77,  # higher the tempe, lower the effect of weights
            degree=2,  # the degree of polyfit when detrend

            eps=1e-6,
    ):
        self.use_real_gap = use_real_gap
        self.use_weights = use_weights
        self.temperature = temperature
        self.degree = degree

        self.eps = eps

        self.train_Et = [None] * self.total_segments
        self.train_Br = [None] * self.total_segments
        self.train_M = [None] * self.total_segments
        self.train_W = [None] * self.total_segments
        self.train_S = [None] * self.total_segments

        self.train_trend = [None] * self.total_segments

        self.val_Et = [None] * self.total_used_stock
        self.val_Br = [None] * self.total_used_stock
        self.val_M = [None] * self.total_used_stock
        self.val_W = [None] * self.total_used_stock
        self.val_S = [None] * self.total_used_stock

        self.val_trend = [None] * self.total_used_stock

        count_slicing = 0
        count_detrending = 0
        count_segmenting = 0

        pos = 0
        for chosen in range(self.total_used_stock):
            start = time.time()

            datasetMaker = DataSetMaker(
                self.feasible_series[chosen],
                self.feasible_date[chosen],
            )
            _, _ = datasetMaker.makeSlices(
                window_len=self.window_len,
            )
            end = time.time()
            count_slicing += end - start

            start = time.time()
            de_trended_x, trend_datas = datasetMaker.detrend(
                use_real_gap=use_real_gap,
                use_weights=use_weights,
                temperature=temperature,
                degree=degree,
            )
            de_trended_x[pos] /= de_trended_x.std(axis=0) + eps
            end = time.time()
            count_detrending += end - start

            index_month, index_weekday = datasetMaker.indexDate()

            for i in range(self.num_segment_each_stock[chosen]):
                start = time.time()

                left = self.feasible_train_len[chosen] - self.length_segment * (i + 1)
                right = self.feasible_train_len[chosen] - self.length_segment * i
                self.train_Et[pos] = de_trended_x[left:right]
                self.train_Br[pos] = trend_datas[left:right, 1:]
                self.train_M[pos] = index_month[left:right]
                self.train_W[pos] = index_weekday[left:right]
                self.train_S[pos] = chosen

                self.train_trend[pos] = np.zeros_like(trend_datas[left + self.pred_days:right + self.pred_days, 0])
                for i in range(degree):
                    self.train_trend[pos] += (trend_datas[left + self.pred_days:right + self.pred_days, i + 1] > 0) * 2 ** i

                left = self.feasible_train_len[chosen]
                right = self.feasible_train_len[chosen] + self.valMin - self.pred_days
                self.val_Et[chosen] = de_trended_x[left:right]
                self.val_Br[chosen] = trend_datas[left:right, 1:]
                self.val_M[chosen] = index_month[left:right]
                self.val_W[chosen] = index_weekday[left:right]
                self.val_S[chosen] = chosen

                self.val_trend[chosen] = np.zeros_like(trend_datas[left + self.pred_days:right + self.pred_days, 0])
                for i in range(degree):
                    self.val_trend[chosen] += (trend_datas[left + self.pred_days:right + self.pred_days, i + 1] > 0) * 2 ** i
 
                end = time.time()
                count_segmenting += end - start

                poltRateOfProcess(
                    (chosen, chosen, pos),
                    (self.total_used_stock, self.total_used_stock, self.total_segments),
                    (count_slicing, count_detrending, count_segmenting),
                    ('slicing', 'detrending', 'segmenting'),
                    ('stock', 'stock', 'segment'), True
                )
                pos += 1

    def formDatapairs(
            self,
            #     Et = True,
            #     beta_rest = True,
            #     month_embed = True,
            #     weekday_embed = True,
            #     stock_lora = True,
    ):
        inputs_train = (
            tf.stack(self.train_Et, 0),
            tf.stack(self.train_Br, 0),
            tf.stack(self.train_M, 0),
            tf.stack(self.train_W, 0),
            tf.stack(self.train_S, 0),
        )
        targets_train = tf.stack(self.train_trend, 0)

        inputs_val = (
            tf.stack(self.val_Et, 0),
            tf.stack(self.val_Br, 0),
            tf.stack(self.val_M, 0),
            tf.stack(self.val_W, 0),
            tf.stack(self.val_S, 0),
        )
        targets_val = tf.stack(self.val_trend, 0)

        u, c = np.unique(targets_train, return_counts=True)
        self.train_per = np.round(c / c.sum() * 100,2)
        print(f'training target has:')
        print(f'\ttype 0: {c[0]} samples, {np.round(100 - self.train_per[1:].sum(),2)}%')
        for i in range(1,len(self.train_per)):
            print(f'\ttype {i}: {c[i]} samples, {self.train_per[i]}%')

        u, c = np.unique(targets_val, return_counts=True)
        self.val_per = np.round(c / c.sum() * 100,2)
        print(f'validation target has:')
        print(f'\ttype 0: {c[0]} samples, {np.round(100 - self.val_per[1:].sum(),2)}%')
        for i in range(1,len(self.val_per)):
            print(f'\ttype {i}: {c[i]} samples, {self.val_per[i]}%')
            
        return inputs_train, targets_train, inputs_val, targets_val
