import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import draw

def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    napoved = 1 / (1 + (np.e ** -np.dot(x, theta)))

    return napoved


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """

    m = len(X)
    cost = 0

    for i in range(m):
        cost += y[i] * np.log(h(X[i], theta)) + (1 - y[i]) * np.log(1 - h(X[i], theta))

    cost = - cost / m

    cost += (lambda_ / (2 * m)) * sum(theta) ** 2

    return cost


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    # ... dopolnite (naloga 1, naloga 2)

    grad = []

    m = len(X)

    for j in range(len(theta)):
        sum = 0

        for i in range(m):
            sum += (y[i] - h(X[i], theta)) * X[i][j]

        grad.append(-1 / m * sum + (2 * lambda_) / m * theta[j])

    return np.array(grad)


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)

    dh = 0.0000001

    grad = []

    m = X.shape[0]

    for j in range(len(theta)):

        tmp_theta = theta.copy()
        tmp_theta[j] = tmp_theta[j] + dh

        sum = (cost(tmp_theta, X, y, 0) - cost(theta, X, y, 0)) / dh

        grad.append(sum + 2 * (lambda_ / m) * theta[j])

    return np.array(grad)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1 - p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X), 1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X, y)
    results = [c(x) for x in X]

    return results


# Iz X in y izbrisimo indekse ki se bodo
# v iteraciji i porabili kot testna mnozica
def create_subdata(X, y, idx):

    y_test = []
    x_test = []

    for i in range(len(idx)):
        y_test.append(y[i])
        x_test.append(X[i])
        np.delete(y, i, 0)
        np.delete(X, i, 0)

    return [X, y, x_test, y_test]


def premesaj(x, y):

    idx = np.array(list(range(len(x))))

    A = np.c_[x, y, idx]
    np.random.shuffle(A)

    n = len(A[0])

    return [A[:, 0:(n-3)], A[:, (n-2)], A[:, (n-1)]]


def test_cv(learner, X, y, k=5):
    # ... dopolnite (naloga 3)

    tocnost = 0

    n = int(len(X) / k) * k

    delitveni_faktor = int(n / k)

    for i in range(k):
        # Indeksi testne mnozice
        idx = list(range(delitveni_faktor * i, delitveni_faktor * (i + 1)))

        # Premesamo ucne podatke.
        learn_data = premesaj(X, y)

        # Razbili bomo mnozici X in y.
        subdata = create_subdata(learn_data[0], learn_data[1], idx)

        # Ucenje na podmnozici.
        classifier = learner(subdata[0], subdata[1])
        napovedi = get_predictions(classifier, subdata[2])

        # Izracun natancnosti.
        tocnost += CA(subdata[3], napovedi)

    return tocnost / k


def CA(real, predictions):

    n = len(real)

    correct = 0

    for i in range(n):

        if predictions[i][0] > predictions[i][1]:
            prediction = 0
        else:
            prediction = 1

        if real[i] == prediction:
            correct = correct + 1

    return correct / n

def AUC(real, predictions):
    # ... dopolnite (dodatna naloga)
    pass


def get_predictions(classifier, x_test):
    return [classifier(x) for x in x_test]


if __name__ == "__main__":
    # Primer uporabe

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y)  # dobimo model

    p = test_cv(learner, X, y)

    napovedi = get_predictions(classifier, X)
    print(test_cv(learner, X, y))

    # draw.draw_decision(X, y, classifier, 0, 1)
