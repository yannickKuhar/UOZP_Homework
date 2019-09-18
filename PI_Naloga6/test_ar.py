import unittest
from ar import AssociationRules, Answers

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


class AssociationRulesTest(unittest.TestCase):

    def setUp(self):
        self.ar = AssociationRules(market_basket)

    def test_frequency(self):
        self.assertEqual(self.ar.frequency({"mleko"}), 4)
        self.assertEqual(self.ar.frequency({"mleko", "kruh"}), 3)
        self.assertEqual(self.ar.frequency({"mleko", "kruh", "sir"}), 2)
        self.assertEqual(self.ar.frequency(
            {"mleko", "kruh", "sir", "zelenjava"}), 1)

    def test_apriori(self):
        res = [
            {"kruh"}, {"zelenjava"}, {"sir"},
            {"kruh", "sir"}, {"zelenjava", "sir"}
        ]

        for s in self.ar.apriori(minsupp=0.6):
            self.assertIn(s, res)
            res.remove(s)

        self.assertListEqual(res, [])

    def test_support(self):
        self.assertEqual(self.ar.support({"mleko"}, {"sir", "kruh"}), 0.2)
        self.assertEqual(self.ar.support({"zelenjava"}, {"sir"}), 0.6)

    def test_confidence(self):
        self.assertEqual(self.ar.confidence({"mleko"}, {"sir", "kruh"}), 0.5)
        self.assertEqual(self.ar.confidence({"zelenjava"}, {"sir"}), 1)

    def test_rules(self):
        res = [
            ({'zelenjava'}, {'kruh'}),
            ({'zelenjava'}, {'sir'}),
            ({'sir'}, {'zelenjava'}),
            ({'kruh'}, {'sir'}),
            ({'sir'}, {'kruh'}),
            ({'zelenjava'}, {'kruh', 'sir'}),
            ({'kruh', 'zelenjava'}, {'sir'}),
            ({'zelenjava', 'sir'}, {'kruh'}),
            ({'kruh', 'sir'}, {'zelenjava'})
        ]

        for s in self.ar.get_rules(minsupp=0.4, minconf=0.6):
            self.assertIn(s, res)
            res.remove(s)

        self.assertListEqual(res, [])


class AnswersTest(unittest.TestCase):

    def setUp(self):
        self.answer = Answers("github.json")

    def test_answer_1(self):
        answer = self.answer.answer_1()
        assert type(answer) is int

    def test_answer_2(self):
        answer = self.answer.answer_2()
        assert type(answer) is float

    def test_answer_3(self):
        answer = self.answer.answer_3()
        assert type(answer) is set or type(answer) is frozenset

    def test_answer_4(self):
        answer = self.answer.answer_4()
        assert type(answer) is set or type(answer) is frozenset

    def test_answer_5(self):
        answer = self.answer.answer_5()
        assert type(answer) is float

    def test_answer_6(self):
        answer = self.answer.answer_6()
        assert type(answer) is float

    def test_answer_7(self):
        answer = self.answer.answer_7()
        assert type(answer) is tuple
        assert type(answer[0]) is set or type(answer[0]) is frozenset
        assert type(answer[1]) is set or type(answer[1]) is frozenset

    def test_answer_8(self):
        answer = self.answer.answer_8()
        assert type(answer) is int

    def test_answer_9(self):
        answer = self.answer.answer_9()
        assert type(answer) is set or type(answer) is frozenset

    def test_answer_10(self):
        answer = self.answer.answer_10()
        assert type(answer) is tuple
        assert type(answer[0]) is set or type(answer[0]) is frozenset
        assert type(answer[1]) is set or type(answer[1]) is frozenset
