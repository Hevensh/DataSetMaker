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
            process_bar=True,
    ):
        self.process_bar = process_bar
        
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
            self.val_length[chosen] = (self.seriesDate.shape[0] - self.train_length[chosen])

            if self.process_bar:
                end = time.time()
                poltRateOfProcess(chosen, self.total_num_stock, end - start, 'loading')

    def chooseFeasibleData(
            self,
            window_len: int = 32,  # window length for segment
            pred_days: int = 16,  # predict length for segment

            min_num_segment=1,  # the min num of segments can be sliced in one stock
            length_segment=2000,  # the length of each segment
    ):
        self.window_len = window_len
        self.pred_days = pred_days
        
        self.min_num_segment = min_num_segment
        self.length_segment = length_segment

        self.feasible_index = self.train_length > (length_segment * min_num_segment + self.window_len)

        self.feasible_train_len = self.train_length[self.feasible_index] - self.window_len + 1
        self.feasible_val_len = self.val_length[self.feasible_index] - self.pred_days

        self.num_segment_each_stock = self.feasible_train_len // length_segment
        self.total_used_stock = self.num_segment_each_stock.shape[0]
        self.total_segments = self.num_segment_each_stock.sum()

        self.valMin = np.unique(self.feasible_val_len).min() - self.pred_days

        if self.process_bar:
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

            input_data_type=(False, True, True, True, True),
            # choose the type of input data (Yt, Et, Beta, Date, Stock) 
            target_data_type=(False, False, True),
            # choose the type of target data (Yt, Et, Beta) 
    ):
        self.use_real_gap = use_real_gap
        self.use_weights = use_weights
        self.temperature = temperature
        self.degree = degree
        
        self.input_data_type = input_data_type
        self.target_data_type = target_data_type
        
        val_len = self.valMin - self.pred_days
        
        if input_data_type[0]:
            self.train_Yt = np.zeros([self.total_segments, self.length_segment, self.window_len, ])
        if input_data_type[1]:
            self.train_Et = np.zeros([self.total_segments, self.length_segment, self.window_len, ])
        if input_data_type[2]:
            self.train_Beta = np.zeros([self.total_segments, self.length_segment, self.degree + 1, ])
        if input_data_type[3]:
            self.train_M = np.zeros([self.total_segments, self.length_segment, ], np.int32)
            self.train_W = np.zeros([self.total_segments, self.length_segment, ], np.int32)
        if input_data_type[4]:
            self.train_S = np.zeros([self.total_segments, ], np.int32)

        if target_data_type[0]:
            self.train_Yt_target = np.zeros([self.total_segments, self.length_segment, self.pred_days + 1, ])
        if target_data_type[1]:
            self.train_Et_target = np.zeros([self.total_segments, self.length_segment, self.pred_days + 1, ])
        if target_data_type[2]:
            self.train_Beta_target = np.zeros([self.total_segments, self.length_segment, self.degree + 1, ])

        if input_data_type[0]:
            self.val_Yt = np.zeros([self.total_used_stock, val_len, self.window_len, ])
        if input_data_type[1]:
            self.val_Et = np.zeros([self.total_used_stock, val_len, self.window_len, ])
        if input_data_type[2]:
            self.val_Beta = np.zeros([self.total_used_stock, val_len, self.degree + 1, ])
        if input_data_type[3]:
            self.val_M = np.zeros([self.total_used_stock, val_len, ])
            self.val_W = np.zeros([self.total_used_stock, val_len, ])
        if input_data_type[4]:
            self.val_S = np.zeros([self.total_used_stock, ])
    
        if target_data_type[0]:
            self.val_Yt_target = np.zeros([self.total_used_stock, val_len, self.pred_days + 1, ])
        if target_data_type[1]:
            self.val_Et_target = np.zeros([self.total_used_stock, val_len, self.pred_days + 1, ])
        if target_data_type[2]:
            self.val_Beta_target = np.zeros([self.total_used_stock, val_len, self.degree + 1, ])
            
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
            slices_x = datasetMaker.makeSlices(
                window_len=self.window_len,
            )
            end = time.time()
            count_slicing += end - start

            if input_data_type[1:3].count(True) + target_data_type[1:].count(True):
                start = time.time()
                de_trended_x, trend_datas = datasetMaker.detrend(
                    use_real_gap=use_real_gap,
                    use_weights=use_weights,
                    temperature=temperature,
                    degree=degree,
                )
                end = time.time()
                count_detrending += end - start

            start = time.time()

            left = self.feasible_train_len[chosen]
            right = self.feasible_train_len[chosen] + val_len
            if input_data_type[0]:
                self.val_Yt[chosen] = slices_x[left:right]
            if input_data_type[1]:
                self.val_Et[chosen] = de_trended_x[left:right]
            if input_data_type[2]:
                self.val_Beta[chosen] = trend_datas[left:right]
            if input_data_type[3]:
                index_month, index_weekday = datasetMaker.indexDate()
                self.val_M[chosen] = index_month[left:right]
                self.val_W[chosen] = index_weekday[left:right]
            if input_data_type[4]:
                self.val_S[chosen] = chosen

            if target_data_type[0]:
                self.val_Yt_target[chosen] = slices_x[left + self.pred_days:right + self.pred_days, self.window_len - self.pred_days - 1:]
            if target_data_type[1]:
                self.val_Et_target[chosen] = de_trended_x[left + self.pred_days:right + self.pred_days, self.window_len - self.pred_days - 1:]
            if target_data_type[2]:
                self.val_Beta_target[chosen] = trend_datas[left + self.pred_days:right + self.pred_days]

            for i in range(self.num_segment_each_stock[chosen]):
                left = self.feasible_train_len[chosen] - self.length_segment * (i + 1)
                right = self.feasible_train_len[chosen] - self.length_segment * i
                if input_data_type[0]:
                    self.train_Yt[pos] = slices_x[left:right]
                if input_data_type[1]:
                    self.train_Et[pos] = de_trended_x[left:right]
                if input_data_type[2]:
                    self.train_Beta[pos] = trend_datas[left:right]
                if input_data_type[3]:
                    self.train_M[pos] = index_month[left:right]
                    self.train_W[pos] = index_weekday[left:right]
                if input_data_type[4]:
                    self.train_S[pos] = chosen

                if target_data_type[0]:
                    self.train_Yt_target[pos] = slices_x[left + self.pred_days:right + self.pred_days, self.window_len - self.pred_days - 1:]
                if target_data_type[1]:
                    self.train_Et_target[pos] = de_trended_x[left + self.pred_days:right + self.pred_days, self.window_len - self.pred_days - 1:]
                if target_data_type[2]:
                    self.train_Beta_target[pos] = trend_datas[left + self.pred_days:right + self.pred_days]

                end = time.time()
                count_segmenting += end - start

                if self.process_bar:
                    if input_data_type[1:3].count(True) + target_data_type[1:].count(True):
                        poltRateOfProcess(
                            (chosen, chosen, pos),
                            (self.total_used_stock, self.total_used_stock, self.total_segments),
                            (count_slicing, count_detrending, count_segmenting),
                            ('slicing', 'detrending', 'segmenting'),
                            ('stock', 'stock', 'segment'), True
                        )
                    else:
                        poltRateOfProcess(
                            (chosen, pos),
                            (self.total_used_stock, self.total_segments),
                            (count_slicing, count_segmenting),
                            ('slicing', 'segmenting'),
                            ('stock', 'segment'), True
                        )
                pos += 1

    def formDatapairs(
            self,
            eps=1e-6,
    ):
        inputs_train = (
            tf.cast(self.train_Et / (self.train_Et.std(axis=-1, keepdims=True) + eps), tf.float32),
            tf.cast(self.train_Beta[:, :, 1:], tf.float32),
            tf.cast(self.train_M, tf.int32),
            tf.cast(self.train_W, tf.int32),
            tf.cast(self.train_S, tf.int32),
        )
        targets_train = np.zeros_like(self.train_Beta_target[:, :, 1], np.int32)
        for i in range(self.degree):
            targets_train += (self.train_Beta_target[:, :, i + 1] > 0) * 2 ** i
        targets_train = tf.cast(targets_train, tf.int32)

        inputs_val = (
            tf.cast(self.val_Et / (self.val_Et.std(axis=-1, keepdims=True) + eps), tf.float32),
            tf.cast(self.val_Beta[:, :, 1:], tf.float32),
            tf.cast(self.val_M, tf.int32),
            tf.cast(self.val_W, tf.int32),
            tf.cast(self.val_S, tf.int32),
        )
        targets_val = np.zeros_like(self.val_Beta_target[:, :, 1], np.int32)
        for i in range(self.degree):
            targets_val += (self.val_Beta_target[:, :, i + 1] > 0) * 2 ** i
        targets_val = tf.cast(targets_val, tf.int32)

        if self.process_bar:
            u, c = np.unique(targets_train, return_counts=True)
            self.train_per = np.round(c / c.sum() * 100, 2)
            print(f'training target has:')
            print(f'\ttype 0: {c[0]} samples, {np.round(100 - self.train_per[1:].sum(), 2)}%')
            for i in range(1, len(self.train_per)):
                print(f'\ttype {i}: {c[i]} samples, {self.train_per[i]}%')
    
            u, c = np.unique(targets_val, return_counts=True)
            self.val_per = np.round(c / c.sum() * 100, 2)
            print(f'validation target has:')
            print(f'\ttype 0: {c[0]} samples, {np.round(100 - self.val_per[1:].sum(), 2)}%')
            for i in range(1, len(self.val_per)):
                print(f'\ttype {i}: {c[i]} samples, {self.val_per[i]}%')

        return inputs_train, targets_train, inputs_val, targets_val
