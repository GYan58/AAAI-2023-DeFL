from Settings import *
from Utils import *
from Aggregations import *


def replaceLayer(GParas,BPara,Layer):
    Res = []
    for i in range(len(GParas)):
        ParaNow = cp.deepcopy(GParas[i])
        
        if Layer != "Full" and Layer != "No":
            C = 0
            for ky in BPara.keys():
                C += 1
                if C == Layer:
                    ParaNow[ky] = cp.deepcopy(BPara[ky])
        
        Res.append(ParaNow)
    return Res


def avgParas(Paras):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = 1 / len(Paras)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res
    

def wavgParas(Paras,Lens):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = Lens[i] / np.sum(Lens)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res


def minusParas(Para1, Multi, Para2):
    Res = cp.deepcopy(Para1)
    for ky in Res.keys():
        Res[ky] = Para1[ky] - Para2[ky] * Multi
    return Res
    

def getGrad(P1, P2):
    Res = cp.deepcopy(P1)
    # print(Res.keys())
    for ky in Res.keys():
        if "weight" in ky or "bias" in ky:
            Res[ky] = P1[ky] - P2[ky]
        else:
            Res[ky] -= Res[ky]
    return Res


def getDirc(Paras, RePara, Pt=0):
    RPara = cp.deepcopy(RePara)
    APara = avgParas(Paras)
    Grad = getGrad(APara, RPara)
    Kys = Grad.keys()
    Direction = cp.deepcopy(RPara)
    Ths = 0
    Num0 = 0
    NumS = 0
    NumB = 0
    for ky in Kys:
        GParas = Grad[ky].cpu().detach().numpy()
        SParas = np.sign((np.abs(GParas) > Ths) * GParas)
        IsFloat = 0
        if type(SParas) == type(np.float32(1.0)):
            IsFloat = 1
        if type(SParas) == type(np.float64(1.0)):
            IsFloat = 1
        if IsFloat == 0:
            Direction[ky] = torch.from_numpy(SParas).to(device)
        else:
            Grad[ky] -= Grad[ky]
        
        Num0 += np.sum(SParas == 0)
        NumS += np.sum(SParas == -1)
        NumB += np.sum(SParas == 1)

    P0 = str(int(NumS / (NumS + Num0 + NumB) * 10000) / 100) + "%"
    P1 = str(int(Num0 / (NumS + Num0 + NumB) * 10000) / 100) + "%"
    P2 = str(int(NumB / (NumS + Num0 + NumB) * 10000) / 100) + "%"
    return Direction


def getcosSim(w0, w1):
    Kys = w0.keys()
    Norm0 = 0.000001
    Norm1 = 0.000001
    Dots = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V0 = w0[ky].cpu()
            V1 = w1[ky].cpu()
            Norm0 += torch.norm(V0) ** 2
            Norm1 += torch.norm(V1) ** 2
            Dots += torch.sum(torch.mul(V0, V1))
    Sim = Dots / np.sqrt(Norm0 * Norm1)
    return Sim


def getDist(w0, w1):
    Dist = 0
    Kys = w0.keys()
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            Dist += np.linalg.norm(w0[ky].cpu().detach().numpy() - w1[ky].cpu().detach().numpy()) ** 2
    Dist = np.sqrt(Dist)
    return Dist


def genParas(Para, Epsi, N):
    Paras = []
    NumP = 0
    Kys = Para.keys()
    for ky in Kys:
        GPara = Para[ky].cpu().detach().numpy()
        Shape = GPara.shape
        Mult = 1
        L = len(Shape)
        for i in range(L):
            Mult *= Shape[i]
        NumP += Mult

    Gap = np.sqrt(Epsi ** 2 / NumP) / N
    for i in range(N):
        NPara = cp.deepcopy(Para)
        for ky in Kys:
            GPara = Para[ky].cpu().detach().numpy() + i * Gap
            IsFloat = 0
            if type(GPara) == type(np.float64(1.0)):
                IsFloat = 1
            if type(GPara) == type(np.float32(1.0)):
                IsFloat = 1
            if IsFloat == 0:
                NPara[ky] = torch.from_numpy(GPara).to(device)
        Paras.append(NPara)
    return Paras


def attkKrum(RePara, KParas, EParas, GNum, AParas):
    if GNum <= 1:
        return [RePara]
    
    Direction = getDirc(KParas, RePara, 1)
    GoalIDs = []
    FindPara = None
    FindLam = 0.1
    Stop = False
    C = 0
    while Stop == False:
        RPara = cp.deepcopy(RePara)
        NPara = minusParas(RPara, FindLam, Direction)
        AParas = []
        Goals = []
        for i in range(GNum):
            AParas.append(NPara)
            Goals.append(i)
        AParas += EParas
        
        _, ID = AggMKrum(AParas, len(Goals))
        
        if ID in Goals:
            Stop = True
            FindPara = NPara
        else:
            FindLam = FindLam * 0.5

        if FindLam < 0.00001:
            Stop = True
            FindPara = NPara
        C += 1

    AttParas = []
    for i in range(GNum):
        AttParas.append(FindPara)
    return AttParas


def attkPKrum(RePara, KParas, EParas, GNum, AParas):
    if GNum <= 1:
        return [RePara]
    
    Direction = getDirc(KParas, RePara, 1)
    GoalIDs = []
    FindPara = None
    FindLam = 0.1
    Stop = False
    C = 0
    while Stop == False:
        RPara = cp.deepcopy(RePara)
        NPara = minusParas(RPara, FindLam, Direction)
        AParas = [NPara]
        Goals = [0]
        AParas += KParas
        _, ID = AggMKrum(AParas, len(Goals))
        
        if ID in Goals:
            Stop = True
            FindPara = NPara
        else:
            FindLam = FindLam * 0.5

        if FindLam < 0.00001:
            Stop = True
            FindPara = NPara
        C += 1

    AttParas = []
    for i in range(GNum):
        AttParas.append(FindPara)
    return AttParas


def attkMinMax(RePara, KParas, GNum, AParas):
    if GNum <= 1:
        return [RePara]

    RPara = cp.deepcopy(RePara)
    Direction = getDirc(KParas, RePara, 1)
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RPara))

    AGD = avgParas(Grads)
    Dists = defaultdict(dict)
    MaxDist = -1
    N = len(KParas)
    for i in range(N):
        G1 = Grads[i]
        for j in range(i, N):
            G2 = Grads[j]
            dist = getDist(G1, G2)
            if dist > MaxDist:
                MaxDist = dist

    Gamma = 0.1
    Stop = False
    FindGrad = None
    while Stop == False:
        NGrad = minusParas(AGD, Gamma, Direction)
        Maxdist = -1
        for i in range(N):
            dist = getDist(NGrad, Grads[i])
            if dist > Maxdist:
                Maxdist = dist

        if Maxdist < MaxDist or Gamma < 0.00001:
            Stop = True
            FindGrad = NGrad
        else:
            Gamma = Gamma * 0.5

    AttParas = []
    FindPara = minusParas(RPara, -1, FindGrad)
    for i in range(GNum):
        AttParas.append(FindPara)

    return AttParas


def attkMinSum(RePara, KParas, GNum, AParas):
    if GNum <= 1:
        return [RePara]

    RPara = cp.deepcopy(RePara)
    Direction = getDirc(KParas, RePara, 1)
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RPara))
    AGD = avgParas(Grads)

    Dists = defaultdict(dict)
    N = len(KParas)
    MaxDist = -1
    for i in range(N):
        G1 = Grads[i]
        GetDist = 0
        for j in range(N):
            G2 = Grads[j]
            dist = getDist(G1, G2)
            GetDist += dist

        if GetDist > MaxDist:
            MaxDist = GetDist

    Gamma = 0.1
    Stop = False
    FindGrad = None
    while Stop == False:
        NGrad = minusParas(AGD, Gamma, Direction)
        Maxdist = 0
        for i in range(N):
            dist = getDist(NGrad, Grads[i])
            Maxdist += dist
        
        if Maxdist < MaxDist or Gamma < 0.00001:
            Stop = True
            FindGrad = NGrad
        else:
            Gamma = Gamma * 0.5

    AttParas = []
    FindPara = minusParas(RPara, -1, FindGrad)
    for i in range(GNum):
        AttParas.append(FindPara)
    return AttParas


def attkLie(RePara, KParas, GNum, ANum, AParas):
    if GNum <= 1:
        return [RePara]
 
    N = ANum
    M = int(N * 0.25)
    S = int(N / 2 + 1) - M
    Z = 5 * st.norm.ppf((N - M - S) / (N - M)) + 0.01
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RePara))

    AGD = avgParas(Grads)
    Direction = getDirc(KParas, RePara, 1)
    
    FGrad = cp.deepcopy(AGD)
    Kys = KParas[0].keys()
    for ky in Kys:
        Gs = []
        for i in range(len(Grads)):
            grad = Grads[i][ky].cpu().detach().numpy()
            Gs.append(grad)

        Mu = np.mean(Gs, axis=0)
        Std = np.std(Gs, axis=0)
        Dirc = Direction[ky].cpu().detach().numpy()
        Res = Mu + Z * Std * (Dirc < 0) - Z * Std * (Dirc > 0)
        FGrad[ky] = torch.from_numpy(Res).to(device)

    AttParas = []
    FPara = minusParas(RePara, -1, FGrad)
    for i in range(GNum):
        AttParas.append(FPara)

    return AttParas
    


