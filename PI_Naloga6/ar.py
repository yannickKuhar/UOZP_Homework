import json
import itertools as it

def read_transactions(file, min_lang=2):
    """
    Read the given json file. Return sets of languages used in repos
    with at least min_lang number of languages
    :param file: json file location
    :param min_lang: minimum number of languages
    :return:
    """
    with open(file) as f:
        data = json.load(f)

    transactions = []
    for repo in data:
        if len(repo['language']) >= min_lang:
            transactions.append(set(l['name'] for l in repo['language']))

    '''
    for t in transactions:
        print(t)
    '''

    return transactions


class AssociationRules:
    """
    Implements Apriori Algorithm and Association Rules
    """
    def __init__(self, transactions):
        """
        :param transactions: list of sets, every set represents one transaction
        """
        self.transactions = transactions

    def frequency(self, items):
        """
        Calculate the frequency for the given set  as an subset of sets
        in transactions
        :param items: set of items
        :return: frequency for the given set of items
        """
        count = 0

        for t in self.transactions:

            if items <= t:
                count = count + 1

        return count

    def get_single_consequents(self, left_side):

        consequents = []

        singles = self.get_singles()
        singles = list(singles[0].union(*singles))

        for s in singles:

            if s not in left_side:
                consequents.append(s)

        return consequents


    def get_singles(self):

        ck = []

        for t in self.transactions:
            for item in t:

                tmp = set()
                tmp.add(item)

                if tmp not in ck:
                    ck.append(tmp)
        return ck

    def get_singles_from_set(self, myset):

        ck = []

        for t in myset:

            tmp = set()
            tmp.add(t)

            if tmp not in ck:
                ck.append(tmp)

        return ck

    # Zdruzuemo samo tiste ki imajo razlicen zadni e.
    def get_candidates(self, ck):

        n = len(ck[0])
        candidates = []

        for i in range(0, len(ck)):
            for j in range(i + 1, len(ck)):

                a = list(ck[i])
                b = list(ck[j])

                a.sort()
                b.sort()

                if n == 1 and a[0] != b[0]:
                    a.append(b[0])

                    if set(a) not in candidates:
                        candidates.append(set(a))

                elif a[0:n - 2] == b[0:n - 2] and a[n - 1] != b[n - 1]:
                    a.append(b[n - 1])

                    if set(a) not in candidates:
                        candidates.append(set(a))

        return candidates

    def num_kplus_size(self, k):

        count = 0

        for t in self.transactions:

            if len(t) >= k:
                count = count + 1

        return count

    def apriori(self, minsupp=0.6):
        """
        Execute the Apriori algorithm and return a list of sets
        with support grater then or equal to minsupp
        :param minsupp: minimal support
        :return: list of sets
        """

        k = 1

        fk = []
        ck = self.get_singles()

        while len(ck) > 0:

            k = k + 1

            tmp = []

            for c in ck:
                if self.frequency(c) / len(self.transactions) >= minsupp and c not in fk:
                    tmp.append(c)

            if len(tmp) == 0:
                break

            ck = self.get_candidates(tmp)
            fk.extend(tmp)

        return list(fk)

    def support(self, left, right):
        """
        Calculate the support of the given rule
        :param left: set of items
        :param right: set of items
        :return: support
        """
        return self.frequency(left.union(right)) / len(self.transactions)

    def confidence(self, left, right):
        """
        Calculate the confidence of the given rule
        :param left: set of items
        :param right: set of items
        :return: confidence
        """
        return self.frequency(left.union(right)) / self.frequency(left)

    '''
    def get_rules_req(self, left, right, result, minconf, n, last_added, level):

        if (set(left), set(right)) in result:
            right.remove(last_added)
            return

        for i in range(0, len(left)):

            right.append(left[i])

            last_added = left[i]

            if level >= len(right) and len(last_added) != 0:


            tmp_left = left[:i] + left[i + 1:]
            tmp_left.sort()

            if self.confidence(set(tmp_left), set(right)) >= minconf and len(right) < n and len(set(tmp_left).intersection(set(right))) == 0:

                result.append((set(tmp_left), set(right)))
                result.append((set(right), set(tmp_left)))

                self.get_rules_req(tmp_left, right, result, minconf, n, last_added, level)
    '''

    def generate_rules(self, fk, H, rules, minconf):

        k = len(fk)
        m = len(H)

        if k > m + 1:

            Hm = self.get_candidates(H)

            for hm in Hm:

                conf = self.confidence(fk, fk.difference(hm))

                if conf >= minconf:
                    rules.append( (fk.difference(hm), hm) )
                else:
                    Hm.remove(hm)

            self.generate_rules(fk, Hm, rules, minconf)

    def get_rules(self, minsupp=0.4, minconf=0.0):

        fk = self.apriori(minsupp)

        rules = []

        for f in fk:
            if len(f) >= 2:
                H1 = self.get_single_consequents(f)
                self.generate_rules(f, H1, rules, minconf)

        return rules

    '''
    def get_rules(self, minsupp=0.4, minconf=0.8):
        """
        Generates association rules and return a list of pairs of sets
        with support grater then or equal to minsupp and confidence grater
        then or equal to minconf
        :param minsupp: minimal support
        :param minconf: minimal confidence
        :return: list of pairs of sets
        """

        # Pridobimo pogoste nabore.
        pn = self.apriori(minsupp)

        rules = []

        for i in pn:
            for j in pn:

                if i != j:

                    conf = self.confidence(i, j)
                    supp = self.support(i, j)

                    if conf >= minconf and len(i.intersection(j)) == 0 and supp >= minsupp:
                        rules.append((set(sorted(i)), set(sorted(j))))

        return rules
        '''

class Answers:
    def __init__(self, file_name):
        """
        Initialize the class
        :param file_name: date file location
        """
        self.ar = AssociationRules(read_transactions(file_name, 5))

    def answer_1(self):
        """
        Pri koliko projektih se uporablja programski jezik Python?
        :return: int
        """
        return self.ar.frequency({"Python"})

    def answer_2(self):
        """
        Kakšen je delež projektov, ki uporablja Python, Shell in C med
        projekti, ki uporabljajo vsaj pet programskih jezikov?
        :return: float
        """
        return self.ar.frequency({"Python", "Shell", "C"}) / self.ar.num_kplus_size(5)

    def answer_3(self):
        """
        Kateri trije programski jeziki se skupaj uporabljajo
        največkrat uporabljajo?
        :return: set
        """
        myrange = [x * 0.10 for x in list(range(1, 10))]
        myrange.reverse()

        for i in myrange:

            search_set = self.ar.apriori(i)

            for s in search_set:

                if len(s) == 3:
                    return s

    def answer_4(self):
        """
        Katera je najpogostejša peterka uporabljanih jezikov?
        :return: set
        """

        max_frq = 0
        max_five = {}

        for t in self.ar.transactions:
            if len(t) == 5:
                if self.ar.frequency(t) > max_frq:
                    max_frq = self.ar.frequency(t)
                    max_five = t

        return max_five

    def answer_5(self):
        """
        Kakšno podporo ima povezovalno pravilo C, C++ --> Makefile?
        :return: float
        """
        return self.ar.confidence({"C", "C++"}, {"Makefile"})

    def answer_6(self):
        """
        Kakšna je ocenjeno zaupanje v pravilo Makefile, Python --> Shell, C
        med projekti, ki uporabljajo vsaj pet jezikov?
        :return: float
        """

        five_or_more = []

        for t in self.ar.transactions:
            if len(t) >= 5:
                five_or_more.append(t)

        ar2 = AssociationRules(five_or_more)

        return ar2.confidence({"Makefile", "Python"}, {"Shell", "C"})

    def answer_7(self):
        """
        Kateremu povezovalnemu pravilu, kjer iz uporabe dveh jezikov
        sklepamo na uporabo tretjega, lahko najbolj zaupamo? Osredotoči
        se samo programske jezike, ki so uporabljani v vsaj 10 % projektov.
        :return: tuple of two sets
        """

        rules = self.ar.get_rules(0.1, 0.9)

        for r in rules:
            if len(r[0]) == 2 and len(r[1]) == 1:
                return r


    def answer_8(self):
        """
       Koliko povezovalnih pravil iz vprašanja 7 ima zaupanje vsaj 0.5?
        :return: int
        """
        return 0

    def answer_9(self):
        """
        Če programer že pozna Python, katera druga dva jezika bo še
        glede na podatke najverjetneje uporabljal?
        :return: set
        """

        rules = self.ar.get_rules()

        # for r in rules:
        return {"a", "b"}

    def answer_10(self):
        """
        Kateremu povezovalnemu pravilu, ki iz souporabe treh jezikov
        sklepa na uporabo četrtega, lahko najbolj zaupamo? Upoštevaj
        samo jezike, ki so uporabljani v vsaj 5 % projektov. Med
        programskimi jeziki izloči C ali C++.
        :return: tuple of two sets
        """
        return ({"a", "b", "c"}, {"d"})


def grq(l, r, res, n):

    if len(r) == 0:
        return

    for i in range(0, len(r)):

        l.append(r[i])

        tmpl = set()
        tmpl = tmpl.union(*l)

        tmpr = set()
        tmpr = tmpr.union(*r[i + 1:])

        if len(r[i + 1:]) > 0 and len(l) < n:

            res.append((tmpl, tmpr))
            res.append((tmpr, tmpl))

            grq(l, r[i + 1:], res, n)

if __name__ == '__main__':

    market_basket = [
        {"mleko", "kruh"},
        {"mleko", "sir", "zelenjava"},
        {"kruh", "sir", "zelenjava"},
        {"mleko", "kruh", "sir", "zelenjava"},
        {"mleko", "kruh", "sir"},
        {"kruh", "sir"},
        {"sir"},
        {"kruh", "zelenjava", "sir"},
        {"sir", "zelenjava"},
        {"kruh", "sir", "zelenjava"}
    ]

    # ar = AssociationRules(read_transactions("github.json", 5))
    # s = ar.get_singles()

    ar = AssociationRules(market_basket)

    # an = Answers("github.json")
    # print("1: ", an.answer_1())
    # print("2: ", an.answer_2())
    # print("3: ", an.answer_3())
    # print("4: ", an.answer_4())
    # print("5: ", an.answer_5())
    # print("6: ", an.answer_6())
    # print("7: ", an.answer_7())

    # res = []
    #
    # grq([], [{1},{2},{3}], res, 3)
    #
    # print(res)

    '''
    for item_set in ar.apriori(minsupp=0.4):
         print(item_set)
    '''


    for rules in ar.get_rules(minsupp=0.4, minconf=0.6):
        print(rules[0], "-->", rules[1])


