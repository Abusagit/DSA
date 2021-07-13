class RandomVar:
    def __init__(self, args):
        self.probability = args
        self.mean = self._mean()
        self.var = self._var()
        self.stdev = self.var ** 0.5

    def _mean(self):
        m = 0
        for item, prob in self.probability.items():
            m += item * prob
        return m

    def _var(self):
        m2 = 0

        for item, prob in self.probability.items():
            m2 += item ** 2 * prob

        return round(m2 - self.mean ** 2, 4)

    def items(self):
        return self.probability.items()

    def __getitem__(self, item):
        return self.probability.get(item)

    @classmethod
    def covariance(cls, X1, X2):
        Exy = 0
        for item, prob in X1.items():
            tmp = 0
            for item2, prob2 in X2.items():
                tmp += item2 * prob2
            tmp *= item
            tmp *= prob
            Exy += tmp

        return X1.mean * X2.mean - Exy


if __name__ == '__main__':
    X = RandomVar({1: 0.05, 2: 0.15, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.04, 7: 0.16})
    print(X.mean, X.var, X.stdev)
    X2 = RandomVar({1: 0.04, 2: 0.15, 3: 0.3, 4: 0.2, 5: 0.2, 6: 0.04, 7: 0.16})
    # print(RandomVar.covariance(X, X2))
