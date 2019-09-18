import math
from datetime import datetime, timedelta
import csv;

from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse

import scipy.sparse as sp
import numpy as np


#################### LINEARNA REGRESIJA ####################
def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


def hl(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)


def cost_grad_linear(theta, X, y, lambda_):
    #do not regularize the first element
    sx = hl(X, theta)
    j = 0.5*numpy.mean((sx-y)*(sx-y)) + 1/2.*lambda_*theta[1:].dot(theta[1:])/y.shape[0]
    grad = X.T.dot(sx-y)/y.shape[0] + numpy.hstack([[0.],lambda_*theta[1:]])/y.shape[0]
    return j, grad


class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = append_ones(X)

        th = fmin_l_bfgs_b(cost_grad_linear,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_))[0]

        return LinearRegClassifier(th)


class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        return hl(x, self.th)
############################################################


def is_week_end(idx):

    if(idx >= 5):
        return 1

    return 0


def parse_datetime(datetimestr):

    tmp = datetimestr.split(' ')

    time = tmp[1].split(':')
    time[2] = str(int(float(time[2])))

    if (len(time[2]) == 1):
        time[2] = '0' + time[2]

    tmp[1] = time[0] + ':' + time[1] + ':' + time[2]

    tmp = tmp[0] + ' ' + tmp[1]

    return datetime.strptime(tmp, '%Y-%m-%d %X')


def read_file(file_name):
    f = open(file_name, "r", encoding="latin1")

    datax = []
    datay = []

    for line in csv.reader(f):
        row = line[0].split('\t')

        # id_line, start_time, end_time
        relevant = [row[2], row[6], row[8]]

        # V spodnjem if stavku se zgodi obdelava
        # datetime stringa
        # 2012-01-03 04:12:04.000

        if(relevant[0] != 'Route'):
            dtstart = parse_datetime(relevant[1])
            dtend = parse_datetime(relevant[2])

            # Priprava tocke za dodajo v data
            # vextor x
            tocka = [int(relevant[0]), dtstart.year, dtstart.month, dtstart.day, dtstart.weekday(), is_week_end(dtstart.weekday()), dtstart.hour, dtstart.minute, dtstart.second]

            razred = 0

            # vektor y
            if (dtstart < dtend):
                razred = (dtend - dtstart).seconds / 60
            else:
                razred = (dtstart - dtend).seconds / 60

            datax.append(tocka)
            datay.append(razred)

    return [datax, datay]


# Prebere testni file in sestavi tocke za testiranje modela na podoben nacin kot ucne tocke.
def read_test_file(file_name):
    f = open(file_name, "r", encoding="latin1")

    data = []

    for line in csv.reader(f):
        row = line[0].split('\t')

        # id_line, start_time
        relevant = [row[2], row[6]]

        if (relevant[0] != 'Route'):

            dtstart = parse_datetime(relevant[1])

            tocka = [int(relevant[0]), dtstart.year, dtstart.month, dtstart.day, dtstart.weekday(),
                     is_week_end(dtstart.weekday()), dtstart.hour, dtstart.minute, dtstart.second]

            data.append(tocka)

    return data

def mse(razredi, napovedi, testni_podatki):

    mi = np.mean(razredi)
    sum = 0

    for i in range(len(napovedi)):
        tocka = datetime(testni_podatki[i][1], testni_podatki[i][2], testni_podatki[i][3],
                         testni_podatki[i][6], int(testni_podatki[i][7]), testni_podatki[i][8])
        delta_minut = (parse_datetime(napovedi[i]) - tocka).seconds / 60
        sum += (delta_minut - mi) ** 2

    return  sum / len(napovedi)


def sestavi_rezultat(test_data, linear):
    napovedi = []

    for test in test_data:
        # tocka = [int(relevant[0]), dtstart.year, dtstart.month, dtstart.day, dtstart.weekday(), is_week_end(dtstart.weekday()), dtstart.hour, dtstart.minute, dtstart.second]

        # v minutah
        prediction = linear(test)

        prediction_sec = float(str(prediction - int(prediction))[1:]) * 60
        prediction_min = math.floor(prediction)

        rezultat = datetime(test[1], test[2], test[3], test[6], int(test[7]), test[8]) + timedelta(minutes=prediction_min, seconds=prediction_sec)

        # olepsamo rezultat z niclam
        second_str = str(float(rezultat.second))
        minute_str = str(rezultat.minute)
        hour_str = str(rezultat.hour)
        day_str = str(rezultat.day)
        month_str = str(rezultat.month)
        year_str = str(rezultat.year)

        if (len(second_str) == 3):
            second_str = '0' + second_str

        if (len(minute_str) == 1):
            minute_str = '0' + minute_str

        if (len(hour_str) == 1):
            hour_str = '0' + hour_str

        if (len(day_str) == 1):
            day_str = '0' + day_str

        if (len(month_str) == 1):
            month_str = '0' + month_str

        # format:%Y-%m-%d h:min:sec
        rezultat = year_str + '-' + month_str + '-' + day_str + ' ' + hour_str + ':' + minute_str + ':' + second_str

        napovedi.append(rezultat)

    return napovedi


def main():

    data = read_file("train.csv")
    test_data = read_test_file("test.csv")

    x = np.array(data[0])
    y = np.array(data[1])

    Xsp = scipy.sparse.csr_matrix(x)

    lr = LinearLearner(lambda_=1.)
    linear = lr(Xsp, y)

    napovedi = sestavi_rezultat(test_data, linear)

    print(mse(data[1], napovedi, test_data))

    fp = open('napovedi.txt', 'w')

    for n in napovedi:
        fp.write(n)
        fp.write('\n')


if __name__ == '__main__':
    main()