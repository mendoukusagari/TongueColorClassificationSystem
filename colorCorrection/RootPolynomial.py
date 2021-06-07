import numpy as np

class RootPolynomial:
    def __init__(self,deg):
        self.deg = deg
    def fit(self,x):
        newRootPoly = []
        temp = self.getPoly([['0'], ['1'], ['2']],self.deg,[['0'], ['1'], ['2']])
        for i in x:
            res = self.mulPoly(temp,i)
            newRootPoly.append(res)
        return np.array(newRootPoly)
    def getPoly(self,rgb,deg,temp):
        if deg > 1:
            for i in range(len(temp)):
                for j in range(len(rgb)):
                    index=temp[i]+temp[j]
                    if not(sorted(index) in temp) and (sorted(index).count(sorted(index)[0]) != len(sorted(index))):
                        temp.append(sorted(index))
            return self.getPoly(rgb,deg-1,temp)
        else:
            return temp

    def mulPoly(self,temp, x):
        res = []
        for i in temp:
            t = 1
            root = 0
            for j in i:
                t*=x[int(j)]
                root+=1
            res.append(t**(1/root))
        return res