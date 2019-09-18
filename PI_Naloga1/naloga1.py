import math
import numpy as np
import csv


# Metoda transponira vhodne podatke.
def transponate_data(data):

    list_values = [i for i in data.values()]

    mtx = np.array(list_values)
    mtx = mtx.transpose()

    list_values = mtx.tolist()

    keys = [i for i in data.keys()]

    new_data = {}

    for i in range(len(keys)):
        new_data[keys[i]] = list_values[i]

    return new_data


# Metoda, ki vzame seznam in ga rekurzivno splosci.
def flatten(cluster):

    l = []

    for i in cluster:
        if isinstance(i, str):
            l.append(i)
        else:
            l = l + flatten(i)

    return l                    


# Sesteje 2 seznama stringov npr. ['1', '2'] + ['3', '4'] = ['4.0', '6.0']
def list_sum(list1, list2):
    pairs = list(zip(list1, list2))
    sum = []

    for i in range(len(pairs)):
        sum.append(str(float(pairs[i][0]) + float(pairs[i][1])))

    return sum


# Prebere file, precisti podatke in sestavi seznam data.
def read_file(file_name):
    f = open("eurovision-final.csv", "r", encoding="latin1")

    data = {}

    for line in csv.reader(f):

        list = [e for e in line[16:63]]
        list = ['0.0' if e == '' else e for e in list]

        line[1] = line[1].strip()

        if line[1] in data.keys():
            data[line[1]] = list_sum(data[line[1]], list)
        else:
            data[line[1]] = list

    novi_kljuci = data['Country']
    # novi_kljuci = [i.rstrip for i in novi_kljuci]

    data.pop('Country', None)

    stari_kljuci = [i for i in data.keys()]

    # Transponiramo podatke vendar se kljuci ne ujemajo z vektorji
    data = transponate_data(data)

    # Uskladimo vektorje z ustreznimi kljuci.
    for i in range(len(stari_kljuci)):
        data[novi_kljuci[i]] = data[stari_kljuci[i]]
        del data[stari_kljuci[i]]
        # print(novi_kljuci[i],data[novi_kljuci[i]])

    data['Montenegro '] = list_sum(data['Montenegro '], data['Serbia & Montenegro'])
    del data['Serbia & Montenegro']

    return data


class HierarchicalClustering:

    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]


    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        x = self.data[r1]
        y = self.data[r2]

        return math.sqrt(sum([(float(a) - float(b))**2 for a,b in zip(x, y)]))


    # Glavna metoda za izracun razdalije med clusterji.
    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        # avg. linkage

        global_sum = 0

        l1 = flatten(c1)
        l2 = flatten(c2)

        for i in l1:
            for j in l2:
                global_sum += self.row_distance(i, j)

        return global_sum / (len(l1) * len(l2))
        

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        min = math.inf * 1
        tmp = math.inf * 1
        c1 = None
        c2 = None

        for i in self.clusters:
            for j in self.clusters:

                if j != i:
                    tmp = self.cluster_distance(i, j)

                    if tmp < min:
                        c1 = i
                        c2 = j
                        min = tmp

        return [c1, c2]


    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """
        while len(self.clusters) > 1:

            closest_two = self.closest_clusters()
            self.clusters.pop(self.clusters.index(closest_two[0]))
            self.clusters.pop(self.clusters.index(closest_two[1]))
            self.clusters.append(closest_two)

        # Bug-fix: Celoten seznam je zapakiran v list 1x prevec
        self.clusters = self.clusters[0]

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """

        ########## POMOZNE METODE IZPIS DENDROGRAMA ##########

        def print_dendrogram(L, size):

            # Ali je par in tipa list.
            def is_pair(T):
                return type(T) == list and len(T) == 2

            # Rekurzivno izracuna visino "drevesa".
            def max_height(L):
                if is_pair(L):
                    h = max(max_height(L[0]), max_height(L[1]))
                else:
                    h = len(str(L))
                return h + size

            activeLevels = {}

            # Metoda se sprehaja po "drevesu" in sestavlja vrstice izpisa.
            # in jo izpise.
            def traverse(L, h, isFirst):
                if is_pair(L):
                    traverse(L[0], h - size, 1)
                    s = [' '] * (h - size)
                    s.append('|')
                else:
                    s = list(str(L[0]))
                    s.append(' ')

                while len(s) < h:
                    s.append('-')

                if (isFirst >= 0):
                    s.append(' ')
                    if isFirst:
                        activeLevels[h] = 1
                    else:
                        del activeLevels[h]

                A = list(activeLevels)
                A.sort()
                for i in A:
                    if len(s) < i:
                        while len(s) < i:
                            s.append(' ')
                        s.append('|')

                print(''.join(s))

                if is_pair(L):
                    traverse(L[1], h - size, 0)

            traverse(L, max_height(L), -1)

        ######################################################

        size = 3
        print_dendrogram(self.clusters, size)

def prefinnepref(hc):
    skup1 = ['Croatia ', 'Slovenia ', 'Bosnia and Herzegovina ', 'Macedonia ', 'Montenegro ']
    skup2 = ['Greece ', 'Russia ', 'Poland ', 'Andorra ', 'Portugal ', 'Armenia ', 'Belarus ', 'Hungary ', 'Czech Republic ', 'Moldova ', 'Serbia ', 'Monaco', 'Azerbaijan ', 'San Marino', 'Slovakia ', 'Romania ', 'Albania ', 'Bulgaria ', 'Spain ', 'Cyprus ', 'Israel ']
    skup3 = ['Estonia ', 'Latvia ', 'Lithuania ', 'Ireland ', 'Malta ', 'Iceland ', 'Norway ', 'Denmark ', 'Finland ']
    skup4 = ['France ', 'Germany ', 'Belgium ', 'Netherlands ']

    min = 1000000.0
    max = 0.0

    idx_min = 0
    idx_max = 0

    keys = [i for i in hc.data.keys()]

    for i in skup4:

        for j in range(len(hc.data[i])):

            if float(hc.data[i][j]) < min:
                min = float(hc.data[i][j])
                idx_min = j
            elif float(hc.data[i][j]) >= max:
                max = float(hc.data[i][j])
                idx_max = j

    print('min:', keys[idx_min], 'max:', keys[idx_max])

# Main metoda.
def main():
    DATA_FILE = "eurovision-final.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    # prefinnepref(hc)
    hc.plot_tree()


if __name__ == "__main__":
    main()