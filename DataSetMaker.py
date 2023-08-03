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

        self.series_x = figure_series
        self.series_date = np.double(date_series) / ns2day

    def makeSlices(
            self,
            window_len=32,
    ):
        self.window_len = window_len
        self.embed_date = self.date_series[window_len - 1:]

        data_len = self.series_x.shape[0]

        sample_len = data_len - window_len + 1
        self.sample_len = sample_len

        self.slices_x = np.zeros([sample_len, window_len])
        self.slices_date = np.zeros([sample_len, window_len])

        for i in range(window_len):
            self.slices_x[:, i] = self.series_x[i:i + sample_len]
            self.slices_date[:, i] = self.series_date[i:i + sample_len]
        self.slices_date -= self.slices_date[:, -1:]

        return self.slices_x, self.slices_date

    def detrend(
            self,
            use_real_gap=True,
            use_weights=True,
            temperature=77,  # higher the tempe, lower the effect of weight
            degree=2,
    ):
        self.use_real_gap = use_real_gap
        self.use_weights = use_weights

        self.temperature = temperature
        self.degree = degree

        poly_degree = degree + 1
        sample_len = self.sample_len
        window_len = self.window_len

        if use_real_gap:
            base_pattern = self.slices_date / window_len

            window_pattern = np.zeros([sample_len, poly_degree, window_len])
            for i in range(poly_degree):
                window_pattern[:, i, :] = base_pattern ** i

        else:
            base_pattern = np.arange(1 - window_len, 1) / window_len

            window_pattern = np.zeros([poly_degree, window_len])
            for i in range(poly_degree):
                window_pattern[i] = base_pattern ** i

        if use_weights:
            weight = temperature / (temperature - base_pattern * window_len)
        else:
            weight = np.eye(window_len)

        if use_real_gap:
            trend_datas = np.zeros([sample_len, poly_degree])
            self.trend_of_slices_x = np.zeros_like(self.slices_x)

            for i in range(sample_len):
                temp_weight = np.diag(weight[i])
                moment_estimate_matrix = np.dot(
                    window_pattern[i].T,
                    np.linalg.inv(window_pattern[i] @ temp_weight @ window_pattern[i].T)
                )

                trend_datas[i] = self.slices_x[i] @ temp_weight @ moment_estimate_matrix
                self.trend_of_slices_x[i] = trend_datas[i] @ window_pattern[i]

        else:
            temp_weight = np.diag(weight)
            moment_estimate_matrix = np.dot(
                window_pattern.T,
                np.linalg.inv(window_pattern @ temp_weight @ window_pattern.T)
            )

            trend_datas = self.slices_x @ temp_weight @ moment_estimate_matrix

            self.trend_of_slices_x = trend_datas @ window_pattern
        de_trended_x = self.slices_x - self.trend_of_slices_x

        return de_trended_x, trend_datas

    def indexDate(
            self,
    ):
        index_month = (self.embed_date.dt.month - 1).to_numpy()
        index_weekday = self.embed_date.dt.weekday.to_numpy()
        return index_month, index_weekday

    def takeALook(
            self,
            i=0,
    ):
        if self.use_real_gap:
            temp_pattern = self.base_pattern[i]
        else:
            temp_pattern = self.base_pattern

        plt.plot(temp_pattern, self.slices_x[i], 'b-',
                 temp_pattern, self.self.trend_of_slices_x[i], 'g--', )
        plt.legend(['real', 'trend'])
        plt.title(f'''the {i}-th slice ,
    real gap: {self.use_real_gap} ,
    weighted: {self.use_weights} .''')

    plt.show()
