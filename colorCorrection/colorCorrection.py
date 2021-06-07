from colorCorrection import KPLSRColorCorrection,RootPolynomialColorCorrection,PolynomialColorCorrection,KPLSROColorCorrection
import time

class ColorCorrection:
    def colorCorrectKPLSR(self,input,source,ref,kernel="rbf",z=10,n=4):
        start = time.time()
        self.ccMachine = KPLSRColorCorrection.KPLSRColorCorrection()
        self.ccMachine.train(source,ref,kernel,z,n)
        end = time.time()
        print(f"Runtime of cc is {end - start}")
        return self.ccMachine.predict(input)
    def colorCorrectRPCC(self,input,source,ref,degree="1"):
        start = time.time()
        self.ccMachine = RootPolynomialColorCorrection.RootPolynomialColorCorrection()
        self.ccMachine.train(source,ref,degree)
        end = time.time()
        print(f"Runtime of cc is {end - start}")
        return self.ccMachine.predict(input)
    def colorCorrectPCC(self,input,source,ref):
        start = time.time()
        self.ccMachine = PolynomialColorCorrection.PolynomialColorCorrection()
        self.ccMachine.train(source, ref)
        end = time.time()
        print(f"Runtime of cc is {end - start}")
        return self.ccMachine.predict(input)
    def colorCorrectKPLSRO(self,input,source,ref,n=4):
        start = time.time()
        self.ccMachine = KPLSROColorCorrection.KPLSROColorCorrection()
        kernel_z_pairs = {"sigmoid":range(4,20),"rbf":range(10,20),"chi":range(2,20)}
        self.ccMachine.train(source,ref,kernel_z_pairs)
        end = time.time()
        print(f"Runtime of cc is {end - start}")
        return self.ccMachine.predict(input)
