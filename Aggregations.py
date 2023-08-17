from Settings import *
from Utils import *

def avgParas(Paras):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = 1 / len(Paras)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res


def AggMKrum(Paras, Frac, Num=1):
    N = len(Paras)
    M = N - Frac
    
    if M <= 1:
        M  = N

    Distances = defaultdict(dict)
    Kys = Paras[0].keys()

    for i in range(N):
        Pa1 = Paras[i]
        for j in range(i,N):
            Pa2 = Paras[j]
            distance = 0
            if i != j:
                for ky in Kys:
                    if "weight" in ky or "bias" in ky:
                        distance += np.linalg.norm(Pa1[ky].cpu().detach().numpy() - Pa2[ky].cpu().detach().numpy()) ** 2
                distance = np.sqrt(distance)
            Distances[i][j] = distance
            Distances[j][i] = distance
    
    if Num == 1:
        FindID = -1
        FindVal = pow(10, 20)
        PDict = {}
        for i in range(N):
            Dis = sorted(Distances[i].values())
            SumDis = np.sum(Dis[:M])
            PDict[i] = SumDis
            if FindVal > SumDis:
                FindVal = SumDis
                FindID = i
        return Paras[FindID], FindID
    
    if Num >= 2:
        Dist = cp.deepcopy(Distances)
        PDict = {}
        for i in range(N):
            Dis = sorted(Dist[i].values())
            SumDis = np.sum(Dis[:M])
            PDict[i] = SumDis
        SDict = sorted(PDict.items(), key=lambda x:x[1])
        
        GParas = []
        for i in range(Num):
            Ky = SDict[i][0]
            GParas.append(Paras[Ky])
        return avgParas(GParas), -1



def AggTrimMean(Paras, Frac):
    N = len(Paras)
    K = Frac
    
    if K >= int(N/2):
        K = int(N/2) - 1

    FPara = cp.deepcopy(Paras[0])
    Kys = Paras[0].keys()
    C = 0
    for ky in Kys:
        Ms = []
        for i in range(N):
            Para = Paras[i][ky].cpu().detach().numpy()
            Ms.append(Para)
        SMs = np.sort(Ms,axis=0)
        
        GMs = []
        for i in range(K,N-K):
            GMs.append(SMs[i])

        GetParas = np.mean(GMs,axis=0)
        FPara[ky] = torch.from_numpy(GetParas).to(device)

    return FPara, -1



def wavgParas(Paras,Lens):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = Lens[i] / np.sum(Lens)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res


def minusParas(Para1,Multi,Para2):
    Res = cp.deepcopy(Para1)
    for ky in Res.keys():
        Res[ky] = Para1[ky] - Para2[ky] * Multi
    return Res


def getGrad(P1,P2):
    Res = cp.deepcopy(P1)
    for ky in Res.keys():
        if "weight" in ky or "bias" in ky:
            Res[ky] = P1[ky] - P2[ky]
        else:
            Res[ky] -= Res[ky]
    return Res


def getSim(w0,w1):
    Kys = w0.keys()
    Norm0 = 0
    Norm1 = 0
    Dots = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V0 = w0[ky].cpu()
            V1 = w1[ky].cpu()
            Norm0 += torch.norm(V0) ** 2
            Norm1 += torch.norm(V1) ** 2
            Dots += torch.sum(torch.mul(V0,V1))
    Sim = Dots / np.sqrt(Norm0 * Norm1)

    return Sim


class AFA:
    def __init__(self):
        self.Alphas = {}
        self.Betas = {}
    
    def Add(self,Id):
        self.Alphas[Id] = 0.5
        self.Betas[Id] = 0.5
    
    def AggParas(self,IDs,RPara,Paras,Lens):
        for ky in IDs:
            if ky not in self.Alphas.keys():
                self.Add(ky)
        
        LcGrads = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            LcGrads[Ky] = getGrad(Paras[i],RPara)
            
        Good = []
        Ls = {}
        Pks = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            Good.append(Ky)
            Ls[Ky] = Lens[i]
            Pks[Ky] = self.Alphas[Ky] / (self.Alphas[Ky] + self.Betas[Ky])
        Bad = []
        R = [0]
        Epi = 1
        Step = 2
    
        while len(R) > 0:
            R = []
            GDs = []
            GLs = []
            for ky in Good:
                GDs.append(LcGrads[ky])
                GLs.append(Ls[ky] * Pks[ky])
            
            GR = wavgParas(GDs,GLs)

            Sims = {}
            ASims = []
            for ky in Good:
                sim = getSim(LcGrads[ky],GR)
                Sims[ky] = sim
                ASims.append(sim)
        
            Mu = np.mean(ASims)
            Std = np.std(ASims)
            Med = np.median(ASims)
           
            for ky in Good:
                sim = Sims[ky]
                IAdd = False
                if Mu < Med:
                    if sim < Med - Std * Epi:
                        IAdd = True
                else:
                    if sim > Med + Std * Epi:
                        IAdd = True
                if IAdd:
                    Bad.append(ky)
                    Good = list(set(Good)-set([ky]))
                    R.append(ky)
            
            Epi = Epi + Step  
        
        GDs = []
        GLs = []
        for ky in Good:
            GDs.append(LcGrads[ky])
            GLs.append(Ls[ky] * Pks[ky])
        GRad = wavgParas(GDs,GLs)
    
        Res = minusParas(RPara,-1,GRad)
        
        for ky in Good:
            self.Alphas[ky] += 1
        for ky in Bad:
            self.Betas[ky] += 1
            
        return Res, -1


def wavgParas2(Paras,Lens):
    Res = 0
    for i in range(len(Paras)):
        Pi = Lens[i] / np.sum(Lens)
        Res += np.array(Paras[i]) * Pi
    
    return Res
    
def getSim2(w0,w1):
    Norm0 = np.linalg.norm(w0) ** 2
    Norm1 = np.linalg.norm(w1) ** 2
    Dots = np.dot(w0,w1)
    Sim = Dots / np.sqrt(Norm0 * Norm1)

    return Sim

class VGN:
    def __init__(self):
        self.Alphas = {}
        self.Betas = {}
    
    def Add(self,Id):
        self.Alphas[Id] = 1
        self.Betas[Id] = 1
    
    def AggParas(self,IDs,RPara,Paras,Lens,Bads,CLP=1):
        for ky in IDs:
            if ky not in self.Alphas.keys():
                self.Add(ky)
                
        Ls = {}
        Pks = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            Ls[Ky] = Lens[i]
            Pks[Ky] = self.Alphas[Ky] / (self.Alphas[Ky] + self.Betas[Ky])
        
        GDs = []
        GLs = []
        for i in range(len(Paras)):
            ky = IDs[i]
            if CLP == 1:
                if i not in Bads:
                    GDs.append(Paras[i])
                    GLs.append(Ls[ky] * Pks[ky])
            else:
                GDs.append(Paras[i])
                GLs.append(Ls[ky] * Pks[ky])
        Res = wavgParas(GDs,GLs)
        
        BIDs = []
        for i in range(len(Paras)):
            ky = IDs[i]
            if i in Bads:
                self.Betas[ky] += 1
                BIDs.append(ky)
            else:
                self.Alphas[ky] += 1
            
        return Res, -1
    

def lawavgParas(Paras,Lens):
    Res = cp.deepcopy(Paras[0])
    C = 0
    for ky in Res.keys():
        NLens = []
        for i in range(len(Lens)):
            NLens.append(Lens[i][C])
        C += 1
        Mparas = 0
        for i in range(len(Paras)):
            Pi = NLens[i] / np.sum(NLens)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res
    

def AggTrust(Pure,IDs,RPara,Paras,Lens): 
    PureGrad = getGrad(Pure, RPara)
    LcGrads = []
    for i in range(len(Paras)):
        LcGrads.append(getGrad(Paras[i], RPara))
    
    if len(Paras) != len(Lens):
        exit()
    
    TSc = []
    for i in range(len(Paras)):
        TSc.append(getSim(PureGrad, LcGrads[i]))
        
    Ws = []
    Bad = []
    for i in range(len(Paras)):
        Ky = IDs[i]
        if TSc[i] > 0:
            Ws.append(TSc[i])
        else:
            Ws.append(0.0)
            Bad.append(Ky)
    
    Res = wavgParas(Paras,Ws)
    
    return Res, -1


def TensorDot(T1,T2):
    Dot = 0
    Kys = list(T1.keys())
    for ky in Kys:
        if "weight" or "bias" in ky:
            V1 = T1[ky].reshape(-1).cpu().detach().numpy()
            V2 = T2[ky].reshape(-1).cpu().detach().numpy()
            Res = np.dot(V1,V2)
            Dot += Res
    return Dot


def TensorMinus(T1,Sym,T2):
    Res = cp.deepcopy(T1)
    Kys = list(T1.keys())
    for ky in Kys:
        if "weight" or "bias" in ky:
            V1 = T1[ky].cpu().detach().numpy()
            V2 = T2[ky].cpu().detach().numpy()
            res = V1 - Sym * V2
            Res[ky] = torch.from_numpy(res).to(device)
    return Res

def TensorMly(T,Sym):
    Res = cp.deepcopy(T)
    Kys = list(T.keys())
    for ky in Kys:
        if "weight" or "bias" in ky:
            V = T[ky].cpu().detach().numpy()
            res = V * Sym
            Res[ky] = torch.from_numpy(res).to(device)
    return Res


def TensorNorm(T):
    Norm = 0
    Kys = list(T.keys())
    for ky in Kys:
        if "weight" or "bias" in ky:
            V = T[ky].cpu().detach().numpy()
            Norm += np.linalg.norm(V) ** 2
    return np.sqrt(Norm)


def TensorAvg(Paras,Lens):
    Res = cp.deepcopy(Paras[0])
    Sum = np.sum(Lens)
    for ky in Res.keys():
        if "weight" or "bias" in ky:
            Mparas = 0
            for i in range(len(Paras)):
                Pi = Lens[i] / Sum
                Mparas += Paras[i][ky] * Pi
            Res[ky] = Mparas
    return Res
    
    
def TensorGVec(T1,Sym,T2):
    Res = []
    Kys = list(T1.keys())
    for ky in Kys:
        if "weight" or "bias" in ky:
            
            V1 = T1[ky].cpu().detach().numpy()
            V2 = T2[ky].cpu().detach().numpy()
            res = V1 - Sym * V2
            Res.append(np.linalg.norm(res))
    return Res
    
    
def getPNorm(V):
    return np.linalg.norm(V)
    

def TensorGVec2(T1,Sym,T2):
    Res = []
    Kys = list(T1.keys())
    C = 0
    for ky in Kys:
        if "weight" in ky:
            V1 = T1[ky].cpu().detach().numpy()
            V2 = T2[ky].cpu().detach().numpy()
            res = V1 - Sym * V2
            res = res.reshape(-1)
            if len(res) > 500:
                gvec = []
                L = int(len(res) / 500)
                gets = []
                for i in range(len(res)):
                    if i % L == 0:
                        gets.append(res[i])
                Res.append(np.abs(gets))
            else:
                Res.append(np.abs(res))
            C += 1
    return Res


def MatMinus(M1,M2):
    Res = []
    for i in range(len(M1)):
        m1 = M1[i]
        m2 = M2[i]
        res = list(np.array(m1) - np.array(m2))
        Res.append(res)
    return Res

def MatAdd(M1,M2):
    Res = []
    for i in range(len(M1)):
        m1 = M1[i]
        m2 = M2[i]
        res = list(np.array(m1) + np.array(m2))
        Res.append(res)
    return Res


def gen0Mat(m,n):
    Res = []
    for i in range(m):
        mat = []
        for j in range(n):
            mat.append(0)
        Res.append(mat)
    return Res

def MatSqrt(M,Sym=1):
    Res = []
    for i in range(len(M)):
        m = abs(M[i])
        res = np.sqrt(m) * Sym
        Res.append(res)
    return Res

def MatEX(M1,M2):
    Res = []
    for i in range(len(M1)):
        V1 = list(M1[i])
        V2 = list(M2[i])
        res = V1 + V2
        Res.append(res)
    return Res


def MatEY(M1,M2):
    Res = []
    for i in range(len(M1)):
        V = list(M1[i])
        res = V
        Res.append(res)
    for i in range(len(M2)):
        V = list(M2[i])
        res = V
        Res.append(res)
    return Res


def Cholesky(matrix):
    w = matrix.shape[0]
    G = np.zeros((w,w))
    for i in range(w):
        G[i,i] = (matrix[i,i] - np.dot(G[i,:i],G[i,:i].T))**0.5
        for j in range(i+1,w):
            G[j,i] = (matrix[j,i] - np.dot(G[j,:i],G[i,:i].T))/G[i,i]
    return G


def Cholesky_plus(matrix):
    w = matrix.shape[0]
    L = np.zeros((w,w))
    for i in range(w):
        L[i,i] = 1
    D = np.zeros((w,w))
    for i in range(w):
        D[i,i] = matrix[i,i] - np.dot(np.dot(L[i,:i],D[:i,:i]),L[i,:i].T)
        for j in range(i+1,w):
            L[j,i] = (matrix[j,i] - np.dot(np.dot(L[j,:i],D[:i,:i]),L[i,:i].T))/D[i,i]
    return L,D



def KMeans(Vs,Iter=100):
    C1 = min(Vs)
    C2 = max(Vs)

    for i in range(Iter):
        Cluster1 = []
        Cluster2 = []
        for v in Vs:
            if abs(v - C1) < abs(v - C2):
                Cluster1.append(v)
            else:
                Cluster2.append(v)
        C1 = np.mean(Cluster1)
        C2 = np.mean(Cluster2)

    Finds = []
    c = 0
    for v in Vs:
        if abs(v - C1) > abs(v - C2):
            Finds.append(c)
        c += 1
    return Finds


class FLDet:
    def __init__(self,Win=10):
        self.Win = Win
        self.DWs = []
        self.DGs = []
        self.LocalUpdates = {}
        self.GlobalUpdate = None
        self.Scores = {}

    def updateDeltas(self,DW,DG):
        self.DWs.append(DW)
        self.DGs.append(DG)
        if len(self.DWs) > self.Win:
            self.DWs = self.DWs[1:]
            self.DGs = self.DGs[1:]

    def getFRes(self,V,Ms,s,q):
        Res = cp.deepcopy(V)
        Kys = list(V.keys())
        for ky in Kys:
            if "weight" or "bias" in ky:
                v = V[ky].cpu().detach().numpy()
                gets = 0
                for j in range(len(Ms)):
                    g = Ms[j][ky].cpu().detach().numpy() * q[j]
                    gets += g
                res = v * s - gets
                Res[ky] = torch.from_numpy(res).to(device)
        return Res

    def getHessian(self,V):
            Mat1 = []
            for i in range(len(self.DWs)):
                V1 = self.DWs[i]
                mat = []
                for j in range(len(self.DWs)):
                    V2 = self.DWs[j]
                    Ele = TensorDot(V1,V2)
                    mat.append(Ele)
                Mat1.append(mat)

            Mat2 = []
            for i in range(len(self.DWs)):
                V1 = self.DWs[i]
                mat = []
                for j in range(len(self.DWs)):
                    V2 = self.DGs[j]
                    Ele = TensorDot(V1,V2)
                    mat.append(Ele)
                Mat2.append(mat)


            Dt = np.diag(np.diag(Mat2))
            Triu = np.triu(Mat2)
            Lt = MatMinus(Mat2,Triu)

            sigma = TensorDot(self.DWs[-1],self.DGs[-1]) / TensorDot(self.DWs[-1],self.DWs[-1])

            Mat3 = sigma * np.array(Mat1) + np.dot(np.dot(Lt,Dt),np.transpose(Lt))

            J,D = Cholesky_plus(Mat3)
            SD = MatSqrt(D)
            J = np.dot(J,SD)
            JT = J.T

            SDt = MatSqrt(Dt)
            NSDt = MatSqrt(Dt,-1)
            SDtL = np.dot(SDt,np.transpose(Lt))

            Mx1 = MatEX(NSDt,SDtL)
            ZMat = gen0Mat(len(SDt),len(SDt[0]))
            Mx2 = MatEX(ZMat,JT)
            M1 = np.array(MatEY(Mx1,Mx2))
            try:
                MR1 = np.linalg.inv(M1)
            except:
                MR1 = np.linalg.pinv(M1)

            ZMat = gen0Mat(len(SDt),len(SDt[0]))
            Mx1 = MatEX(SDt,ZMat)
            Mx2 = MatEX(SDtL,J)
            M2 = np.array(MatEY(Mx1,Mx2))
            try:
                MR2 = np.linalg.inv(M2)
            except:
                MR2 = np.linalg.pinv(M2)

            MR3 = []
            for i in range(len(self.DGs)):
                G = self.DGs[i]
                res = TensorDot(G,V)
                MR3.append(res)
            for i in range(len(self.DWs)):
                W = self.DWs[i]
                res = TensorDot(W,V) * sigma
                MR3.append(res)

            Q1 = np.dot(MR1,MR2)
            Q2 = np.dot(Q1,MR3)
            Q = cp.deepcopy(Q2)
            for i in range(len(self.DWs)):
                Q[-1-i] *= sigma

            Paras = self.DGs + self.DWs
            Res = self.getFRes(V,Paras,sigma,Q)
            return Res

    def detection(self,Wt,LWts,Kys,RW,MNum,BNum):
        DW = TensorMinus(Wt,1,RW)
        G = DW
        DG = None
        if self.GlobalUpdate != None:
            DG = TensorMinus(G,1,self.GlobalUpdate)
        self.GlobalUpdate = cp.deepcopy(G)

        for ky in Kys:
            if ky not in self.Scores.keys():
                self.Scores[ky] = []

        if len(self.DWs) >= 10:
            Scores = []
            Addition = self.getHessian(DW)
            for i in range(len(Kys)):
                ky = Kys[i]
                OldUpdate = cp.deepcopy(G)
                if ky in self.LocalUpdates.keys():
                    OldUpdate = cp.deepcopy(self.LocalUpdates[ky])
                PredUpdate = TensorMinus(OldUpdate,-1,Addition)
                RealUpdate = TensorMinus(LWts[i],1,RW)
                Gap = TensorMinus(PredUpdate,1,RealUpdate)
                score = TensorNorm(Gap)
                Scores.append(score)

            Scores = list(np.array(Scores) / np.sum(Scores))
            for i in range(len(Kys)):
                ky = Kys[i]
                score = Scores[i]
                self.Scores[ky].append(score)

        VScore = []
        for i in range(len(Kys)):
            ky = Kys[i]
            Score = self.Scores[ky]
            if len(Score) > 0:
                score = np.mean(Score[-self.Win:])
                VScore.append(score)

        Bads = []
        if len(VScore) > 10:
            Bads = KMeans(VScore)

        for i in range(len(LWts)):
            ky = Kys[i]
            WNow = LWts[i]
            Gi = TensorMinus(WNow,1,RW)
            self.LocalUpdates[ky] = Gi

        if DG != None:
            self.updateDeltas(DW,DG)

        Bads = list(np.unique(Bads))
        Labels = []
        for ky in Bads:
            Labels.append(Kys[ky])

        TBads = []
        for i in range(MNum):
            TBads.append(i)

        TPR = int(len(set(Bads) & set(TBads)) / MNum * 100) / 100
        FPR = int(len(set(Bads) - set(TBads)) / BNum * 100) / 100

        return Bads, TPR, FPR





