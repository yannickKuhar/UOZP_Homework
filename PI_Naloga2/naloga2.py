import glob
import os.path
import math
import random
import numpy as np


'''
Nabor 20 jezikov:
L ~ latinica
C ~ cirilica
? ~ Neznano

Slovanski   Germanski   Romansk   Ostalo
Slv(L)      Eng(L)      Fra(L)    Swa(?)
Slo(L)      Ger(L)      Esp(L)    Gre(?)
Rus(C)      Swd(L)      Por(L)    Trk(L)
Blg(C)      Fin(L)      Ita(L)    Kkn(?)
Czc(L)      Ice(L)                Chn(?)    
                                  Jpn(?)
'''
########################################################################################################################
#                                     POMOZNE METODE ZA K-MEDIOIDS                                                     #
########################################################################################################################
# Preberemo podatke iz datotek in sestavimo
# slovar oblike {drzava: text}
def read_files():
    corpus = {}

    for file_name in glob.glob('nabor/*'):
        name = os.path.splitext(os.path.basename(file_name))[0]
        text = " ".join([line.strip() for line in open(file_name, "rt", encoding="utf8").readlines()])
        text = text.lower()
        corpus[name] = text

    return corpus

def read_files2():
    corpus = {}

    for file_name in glob.glob('clanki/*'):
        name = os.path.splitext(os.path.basename(file_name))[0]
        text = " ".join([line.strip() for line in open(file_name, "rt", encoding="utf8").readlines()])
        text = text.lower()
        corpus[name] = text

    return corpus

# Naredimo touple zeljene velikosti iz dolocenega besedila
def kmers(s, k=3):
    """Generates k-mers for an input string."""
    for i in range(len(s)-k+1):
        yield s[i:i+k]

def parse_string(besedilo):
    return list(kmers(besedilo, 3))

# Iz besedila sestavimo seznam touplov istolezni drzavi.
def parse_data(corpus):
    return {k: set(kmers(corpus[k], 3)) for k in corpus}


# Izracuna st. pojavov substringa v stringu.
def nmatches(substr, str):
    return str.count(substr) # sum(str[i:].startswith(substr) for i in range(len(str)))

def get_point_from_text(besedilo, subs):
    return [nmatches(sub, besedilo) / len(besedilo) for sub in subs]


# Iz podatkov in corpusa dobimo vektorje pojavitev touplov v besedilu oz.
# nase tocke iz vecdimenzionalnega prostora.
def get_points(data, corpus):
        points = {}

        for name in corpus.keys():
            points[name] = [nmatches(sub, corpus[name]) / len(data[name]) for sub in data[name]]

        return points


# Kosinusna razdalija.
def cos_razdalija(a, b):
    return sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))


# Dobi st komponent najdaljsega vektorja
def get_max_len(points):

    max_len = 0

    for i in points.keys():
        if len(points[i]) > max_len:
            max_len = len(points[i])

    return max_len

########################################################################################################################
#                                            K-MEDIOIDS                                                                #
########################################################################################################################

class KMedioids:

    def __init__(self, points, n_clusters):
        self.vectors = points.copy()
        self.points = points
        self.n_clusters = n_clusters
        self.clusters = []
        self.max_len = get_max_len(points)


    def select_medioids(self):

        medoids = {}

        for i in range(self.n_clusters):
            key = random.choice(list(self.points.keys()))
            medoids[key] = self.points[key]
            self.points.pop(key)

        return medoids


    # Inicializira clusterje kot prazne sezname.
    def init_clusters(self):
        for i in range(self.n_clusters):
            self.clusters.append([])


    # Sprazni seznam clusterjev.
    def clear_cluster(self):
        self.clusters = []


    def arrangement_similarity(self, medioids):

        sum = 0

        medioids_key = list(medioids.keys())
        medioids_val = list(medioids.values())

        for i in range(self.n_clusters):
            for point in self.clusters[i]:
                    sum += cos_razdalija(medioids_val[i], self.vectors[point])

        return sum


    def associate(self, medioids):

        self.clear_cluster()
        self.init_clusters()

        medioids_keys = list(medioids.keys())
        medioids_vals = list(medioids.values())

        # Add medioids to clusters
        for i in range(len(medioids_keys)):
            self.clusters[i].append(medioids_keys[i])

        # Add non medioids to clusters.
        for point in self.points.keys():

            idx = math.inf * 1
            max_sin = 0

            for i in range(len(medioids_keys)):

                sin = cos_razdalija(medioids_vals[i], self.points[point])

                if sin >= max_sin:
                    max_sin = sin
                    idx = i

            self.clusters[idx].append(point)


    def run(self):

        self.init_clusters()
        medioids = self.select_medioids()
        self.associate(medioids)

        medioid_val_tmp = []

        for medioid in medioids.keys():

            medioid_val_tmp = medioids[medioid]

            for point in self.points.keys():

                sim = self.arrangement_similarity(medioids)

                medioids.pop(medioid)
                medioids[point] = self.points[point]

                self.associate(medioids)

                new_sim = self.arrangement_similarity(medioids)

                if new_sim > sim: # Swap back
                    medioids.pop(point)
                    medioids[medioid] = medioid_val_tmp
                    break
########################################################################################################################
#                                             METODA SIHUETE                                                           #
########################################################################################################################

def all_a_distances(km):

    all_a_dist = {}

    for cluster in km.clusters:

        for i in cluster:

            ai = 0

            for j in cluster:

                if i != j:
                    ai += cos_razdalija(km.vectors[i], km.vectors[j])

            ai = ai / len(cluster)
            all_a_dist[i] = ai
    return all_a_dist

def all_b_distances(km):

    bi = {}

    for ci in km.clusters:

        for cj in km.clusters:

            ai = 0

            if ci != cj:

                for xi in ci:
                    for xj in cj:
                        ai += cos_razdalija(km.vectors[xi], km.vectors[xj])

                    ai = ai / len(cj)

                    if xi in bi and ai < bi[xi]:
                        bi[xi] = ai
                    elif xi not in bi:
                        bi[xi] = ai
    return bi


def sihueta_razbitja(km):

    a = all_a_distances(km)
    b = all_b_distances(km)

    si = {}

    for k in a.keys():
        if b[k] == 0 and a[k] == 0:
            si[k] = 0
        else:
            si[k] = (b[k] - a[k]) / max(b[k], a[k])

    sihueta = 0

    for k in si.keys():
        sihueta += si[k]

    return sihueta / len(si.keys())

########################################################################################################################
#                                                HISTOGRAM                                                             #
########################################################################################################################

def histogram(points, n_clusters):

    list_sihuet = []

    for i in range(100):
        tmp_pts = points.copy()
        km = KMedioids(tmp_pts, n_clusters)
        km.run()
        list_sihuet.append(round(sihueta_razbitja(km), 2))

    hist = {}

    list_sihuet.sort()

    for i in list_sihuet:
        hist[i] = hist.get(i, 0) + 1

    return hist

def print_hist_ascii(hist):

    for i in hist.keys():
        print(i, '+' * hist[i])

########################################################################################################################
#                                          NAPOVEDOVANJE JEZIKA                                                        #
########################################################################################################################

def napovej_jezik(besedilo, points):
    kmers_besedila = parse_string(besedilo)
    toka = get_point_from_text(besedilo, kmers_besedila)

    rez = ""
    max_podobnost = 0

    for key in points.keys():

        podobnost = cos_razdalija(toka, points[key])

        if podobnost >= max_podobnost:
            rez = key
            max_podobnost = podobnost

    print(rez, max_podobnost)

########################################################################################################################
#                                                  MAIN                                                                #
########################################################################################################################

def main():

    ''' K-Medioids Algoritem '''
    corpus = read_files()
    data = parse_data(corpus)
    points = get_points(data, corpus)

    '''
    for n in range(3, 11):
        tmp_pts = points.copy()
        km = KMedioids(tmp_pts, n)
        km.run()
        print(n, round(sihueta_razbitja(km), 2))
    '''

    #n_clusters = 10

    # km = KMedioids(points, n_clusters)
    # km.run()

    # for c in km.clusters:
        # print(c)

    ''' Metoda Sihuete '''
    # Zelo neoptimalno potrebuje 1min 30s za izvedbo.
    # hist = histogram(points, n_clusters)
    # print_hist_ascii(hist)

    ''' Napoved Jezika '''
    besedilo1 = "Danes je lep dan, na nebu ni nobenih oblakov.."
    besedilo2 = "Today is a good day, there are no clouds in the sky."
    besedilo3 = "Heute ist ein guter Tag es gibt keine Wolken an dem Himmel."
    besedilo4 = "Dnes je dobrý den, na obloze nejsou žádné mraky."
    besedilo5 = "Aujourd'hui est un bon jour, il n'y a pas de nuages ​​dans le ciel."
    napovej_jezik(besedilo5, points)

    ''' Clanki '''
    
    # n_clusters = 5
    # corpus2 = read_files2()
    # data2 = parse_data(corpus)
    # points2 = get_points(data2, corpus2)

    # for i in points2.keys():
        # print(i, points2[i])

    # hist2 = histogram(points2, n_clusters)
    # print_hist_ascii(hist2)

if __name__ == '__main__':
    main()


