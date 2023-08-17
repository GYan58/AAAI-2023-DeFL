from Settings import *
from Models import Alex as AX
from Models import VGG as VG
from Models import FNN as FN

import warnings
warnings.filterwarnings("ignore")

class CPA:
    def __init__(self, Delta=0.05, Recent = 10):
        self.R = Recent
        self.Round = 0
        self.Threshold = Delta
        self.FGNs = []
        for i in range(20):
            self.FGNs.append(0)
        self.MLim = 5
            
    def Proc(self,Ls,Gs):
        self.Round += 1
        SumL = np.sum(Ls)
        FGN = 0
        for i in range(len(Ls)):
            FGN += Ls[i] / SumL * Gs[i]
        return FGN
    
    def Judge(self,Ls,Gs):
        FGN = self.Proc(Ls,Gs)

        Old = np.mean(self.FGNs[-self.R:]) + 0.00000001
        self.FGNs.append(FGN)
        New = np.mean(self.FGNs[-self.R:])

        Is = 0
        if (New - Old) / Old >= self.Threshold or self.Round <= self.MLim:
            Is = 1

        return Is


def RBudget(Iter,FMY,SMY,CP):
    TotalB = Iter
    LeftB = TotalB - CP * FMY
    LeftR = Iter - CP
    BRecover = (LeftB - LeftR) / (SMY - 1)
    if int(BRecover) < BRecover:
        BRecover = int(BRecover) + 1 + CP
    else:
        BRecover = int(BRecover) + CP
    return BRecover


def toStr(V):
    S = ""
    for v in V:
        S += str(v) +","
    S = S[:-1]
    return S


def load_Model(Type, Name):
    Model = None
    if Type == "fnn":
        if Name == "mnist":
            Model = FN.fnn_mnist()

        if Name == "fmnist":
            Model = FN.fnn_fmnist()

        if Name == "cifar10":
            Model = FN.fnn_cifar10()
            
    if Type == "alex":
        if Name == "mnist":
            Model = AX.alex_mnist()

        if Name == "fmnist":
            Model = AX.alex_fmnist()

        if Name == "cifar10":
            Model = AX.alex_cifar10()
     
    if Type == "vgg":
        if Name == "mnist":
            Model = VG.vgg_mnist()

        if Name == "fmnist":
            Model = VG.vgg_fmnist()

        if Name == "cifar10":
            Model = VG.vgg_cifar10()

    return Model


def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY


def get_mnist():
    data_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY


def get_fmnist():
    data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY


# add Blur
class Addblur(object):

    def __init__(self, blur="Gaussian"):
        self.blur = blur

    def __call__(self, img):
        if self.blur == "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        if self.blur == "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        if self.blur == "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img

# add noise
class AddNoise(object):
    def __init__(self, noise="Gaussian"):
        self.noise = noise
        self.density = 0.8
        self.mean = 0.0
        self.variance = 10.0
        self.amplitude = 10.0

    def __call__(self, img):

        img = np.array(img) 
        h, w, c = img.shape

        if self.noise == "pepper":
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]) 
            mask = np.repeat(mask, c, axis=2)  
            img[mask == 2] = 0 
            img[mask == 1] = 255 

        if self.noise == "Gaussian":
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255 

        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img


class split_image_data(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, dataset, labels, workers, balance=True, isIID=True, alpha=0.0):
        seed = 1234
        Perts = []
        self.Dataset = dataset
        self.Labels = labels 
        self.workers = workers
        self.DirichRVs = []
        self.DirichCount = 0

        if alpha == 0 and not isIID:
            print("* Split Error...")

        if balance:
            for i in range(workers):
                Perts.append(1 / workers)
        else:
            Sum = workers * (workers + 1) / 2
            SProb = 0
            for i in range(workers - 1):
                prob = int((i + 1) / Sum * 10000) / 10000
                SProb += prob
                Perts.append(prob)

            Left = 1 - SProb
            Perts.append(Left)
            bfrac = 0.1 / workers
            for i in range(len(Perts)):
                Perts[i] = Perts[i] * 0.9 + bfrac

        if not isIID and alpha > 0:
            self.partitions = self.__getDirichlet__(labels, Perts, seed, alpha)
        if isIID:
            self.partitions = []
            rng = rd.Random()
            rng.seed(seed)
            data_len = len(labels)
            indexes = loadShuffle(data_len)
            for frac in Perts:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def __getDirichlet__(self, data, psizes, seed, alpha):
        n_nets = len(psizes)
        K = len(np.unique(self.Labels))
        labelList = np.array(data)
        min_size = 0
        N = len(labelList)
        np.random.seed(seed)

        net_dataidx_map = {}
        idx_batch = []
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                self.DirichCount += 1
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        return idx_batch

    def get_splits(self):
        clients_split = []
        for i in range(self.workers):
            IDx = self.partitions[i]
            Ls = self.Labels[IDx]
            Ds = self.Dataset[IDx]

            Xs = []
            Ys = []
            Datas = {}
            for k in range(len(Ls)):
                L = Ls[k]
                D = Ds[k]
                if L not in Datas.keys():
                    Datas[L] = [D]
                else:
                    Datas[L].append(D)

            Kys = list(Datas.keys())
            Kl = len(Kys)
            CT = 0
            k = 0
            while CT < len(Ls):
                Id = Kys[k % Kl]
                k += 1
                if len(Datas[Id]) > 0:
                    Xs.append(Datas[Id][0])
                    Ys.append(Id)
                    Datas[Id] = Datas[Id][1:]
                    CT += 1

            clients_split += [(np.array(Xs), np.array(Ys))]
            del Xs, Ys
            gc.collect()

        n_labels = len(np.unique(self.Labels))

        return clients_split


def get_train_data_transforms(name, aug=False, blur=False, noise=False, normal=False):
    Ts = [transforms.ToPILImage()]
    if name == "mnist" or name == "fmnist":
        Ts.append(transforms.Resize((32, 32)))

    if aug == True and name == "cifar10":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())

    if blur == True:
        Ts.append(Addblur())

    if noise == True:
        Ts.append(AddNoise())

    Ts.append(transforms.ToTensor())

    if normal == True:
        if name == "mnist":
            Ts.append(transforms.Normalize((0.06078,), (0.1957,)))
        if name == "fmnist":
            Ts.append(transforms.Normalize((0.1307,), (0.3081,)))
        if name == "cifar10":
            Ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    return transforms.Compose(Ts)


def get_test_data_transforms(name, normal=False):
    transforms_eval_F = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
    }

    transforms_eval_T = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
    }

    if normal == False:
        return transforms_eval_F[name]
    else:
        return transforms_eval_T[name]
 
 
class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_loaders(Name, n_clients=10, isiid=True, alpha=0.0, aug=False, noise=False, blur=False, normal=False,dshuffle=True, batchsize=128, vpert=0.1):
    TrainX, TrainY, TestX, TestY = [], [], [], []
    if Name == "mnist":
        TrainX, TrainY, TestX, TestY = get_mnist()
    if Name == "fmnist":
        TrainX, TrainY, TestX, TestY = get_fmnist()
    if Name == "cifar10":
        TrainX, TrainY, TestX, TestY = get_cifar10()

    transforms_train = get_train_data_transforms(Name, aug, blur, noise, normal)
    transforms_eval = get_test_data_transforms(Name, normal)

    splits = split_image_data(TrainX, TrainY, n_clients, True, isiid, alpha).get_splits()

    client_loaders = []
    valid_x = []
    valid_y = []
    SumL = 0
    VdL = 0
    DifXs = {}
    DifYs = {}
    for i in range(10):
        DifXs[i] = []
        DifYs[i] = []
    
    for x, y in splits:
        if vpert < 0.5:
            L = int(len(x) * vpert)
            vx = list(x[:L])
            vy = list(y[:L])
            valid_x += vx
            valid_y += vy
            VdL += L
            
            for i in range(len(vy)):
                yi = vy[i]
                xi = vx[i]
                DifXs[yi].append(xi)
                DifYs[yi].append(yi)
        
        client_loaders.append(
            torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batchsize, shuffle=dshuffle))

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(TrainX, TrainY, transforms_eval), batch_size=2000,shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(TestX, TestY, transforms_eval), batch_size=2000,shuffle=False, num_workers=2)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)
    valid_loader = torch.utils.data.DataLoader(CustomImageDataset(valid_x, valid_y, transforms_train), batch_size=batchsize,shuffle=False)
    
    dif_loaders = {}
    for ky in DifXs.keys():
        X = np.array(DifXs[ky])
        Y = np.array(DifYs[ky])
        loader_now = torch.utils.data.DataLoader(CustomImageDataset(X, Y, transforms_train), batch_size=batchsize,shuffle=False)
        dif_loaders[ky] = loader_now
        
    
    stats = {"split": [x.shape[0] for x, y in splits]}

    return client_loaders, train_loader, test_loader, valid_loader, dif_loaders


def avgParas(Paras):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = 1 / len(Paras)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res

def getNorm(P):
    Kys = P.keys()
    Norm = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V = P[ky].cpu()
            Norm += torch.norm(V) ** 2
    Norm = np.sqrt(Norm)
    return Norm

def genNormal(Vec):
    Shape = Vec.shape
    Res = np.random.normal(0,0.1,Shape)
    return Res

def minusParas(P1,Multi,P2):
    Res = cp.deepcopy(P1)
    for ky in Res.keys():
        Res[ky] = P1[ky] - Multi * P2[ky]
    return Res

def scaleParas(P,Fac):
    Res = cp.deepcopy(P)
    for ky in Res.keys():
        Res[ky] = P[ky] * Fac
    return Res

def genNewPara(Paras):
    Res = cp.deepcopy(Paras)
    Kys = Res.keys()
    for ky in Kys:
        Vec = Paras[ky].cpu().detach().numpy()
        NVec = genNormal(Vec)
        Res[ky] = torch.from_numpy(NVec).to(device)
    return Res
    
def sameDirc(G0,G1):
    Kys = G0.keys()
    Res = cp.deepcopy(G1)
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            GD0 = G0[ky].cpu().detach().numpy()
            GD1 = G1[ky].cpu().detach().numpy()
            
            GD1_1 = (GD1 >= 0) * GD1 + (GD1 < 0) * GD1 * -1
            GD1_2 = (GD1 < 0) * GD1 + (GD1 >= 0) * GD1 * -1
            GRes = (GD0 >= 0) * GD1_1 + (GD0 < 0) * GD1_2
            Res[ky] = torch.from_numpy(GRes).to(device)
    return Res


def attkAdaptGen(Model,G0,Gamma=0.001,Loader=None,Eta=0.001):
    Paras = None
    
    V = 1
    count = 0
    while count < V:
        count += 1
        Paras = Model.state_dict()
        NParas = genNewPara(Paras)
        NModel = cp.deepcopy(Model)
    
        AddParas = minusParas(Paras,-Gamma,NParas)
        NModel.load_state_dict(AddParas)
        loss_fn = nn.CrossEntropyLoss()

        Facs = []
        C = 0
        for batch_id, (inputs, targets) in enumerate(Loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = Model(inputs)
            outputs2 = NModel(inputs)
            loss1 = loss_fn(outputs1, targets).item()
            loss2 = loss_fn(outputs2, targets).item()
            Facs.append(loss2 - loss1)
            C += 1
            if C >= 5:
                break
    
        Fac = np.mean(Facs) / Gamma
        Grad = scaleParas(NParas,Fac)
        NewParas = minusParas(Paras,-Eta,Grad)
        ParaNorm = getNorm(NewParas)
        Paras = scaleParas(NewParas,ParaNorm)
        Model.load_state_dict(Paras)
    
    
    GradNorm = getNorm(G0)
    NewParas = scaleParas(Paras,GradNorm)

    return NewParas


def Lorenz(a,b,X):
    return np.exp(-(a-X)/b)

def tangent(a,b,X):
    A = np.exp(-(a-X)/b) * 1 / b
    X1 = X
    Y1 = Lorenz(a,b,X1)

    B2 = Y1 - A * X1
    X2 = -B2 / A
    return X2


def getCosin(V1,V2):
    N1 = np.linalg.norm(V1)
    N2 = np.linalg.norm(V2)
    Dot = np.dot(V1,V2)
    return Dot / N1 / N2 + 1.0


def LorenzThr(Vs):
    Vs = sorted(Vs, reverse=True)
    Fs = []
    Ls = []
    for i in range(len(Vs)):
        if np.isnan(Vs[i]) == False:
            Fs.append(i + 1)
            Ls.append(np.log(Vs[i]))

    Xs = np.array(Ls).reshape((len(Ls),1))
    Ys = np.array(Fs).reshape((len(Fs),1))

    LModel = LinearRegression()
    LModel.fit(Xs,Ys)

    b = LModel.coef_[0][0]
    a = LModel.intercept_[0]

    Cut1 = tangent(a,b,0)
    Cut2 = min(tangent(a,b,Cut1), len(Vs))

    R = (Cut1 + (Cut2 - Cut1) * 0.5) / len(Vs) * 100

    return R


def pDDFs(DDF):
    Res = []

    for i in range(len(DDF)):
        val = DDF[i]
        if val < 0.05:
            val = 0.0
        Res.append(val)

    Res = list(np.array(Res) / np.sum(Res))
    return Res


def WVGNs(VGNs,DDFs):
    Mat = []

    for j in range(len(VGNs)):
        DDF = pDDFs(DDFs[j])
        Row = list(np.array(VGNs[j][0]) * DDF[0])
        for k in range(1, len(VGNs[j])):
            Row += list(np.array(VGNs[j][k]) * DDF[k])
        Mat.append(Row)

    return Mat
    
        

def LWVGNs(VGNs,DDFs):
    Mats = []

    for k in range(len(VGNs[0][0])):
        GMat = []
        for i in range(len(VGNs)):
            gmat = []
            DDF = pDDFs(DDFs[i])
            for j in range(len(VGNs[0])):
                val = VGNs[i][j][k] * DDF[j]
                gmat.append(val)
            GMat.append(gmat)

        Mats.append(GMat)
    
    print(len(Mats),len(Mats[0]),len(Mats[0][0]))

    return Mats
    
    
def LayerVGNs(VGNs):
    Mats = []

    for k in range(len(VGNs[0][0])):
        GMat = []
        for i in range(len(VGNs)):
            gmat = []
            for j in range(len(VGNs[0])):
                val = VGNs[i][j][k]
                gmat.append(val)
            GMat.append(gmat)

        Mats.append(GMat)
    
    print(len(Mats),len(Mats[0]),len(Mats[0][0]))

    return Mats

 
def samLen(X1,X2):
    Y1 = []
    Y2 = []
    for i in range(len(X1)):
        if X1[i] != 0 and X2[i] != 0:
            Y1.append(X1[i])
            Y2.append(X2[i])
        else:
            Y1.append(0)
            Y2.append(0)
    
    return Y1, Y2
    
    
def PCADim(Mat):
    L = min(len(Mat),20)
    modelPCA2 = PCA(n_components=L)
    Xtrans = modelPCA2.fit_transform(Mat)
    return Xtrans
    
def NormVec(V):
    return list(np.array(V) / np.sum(V))
    
def getCov(X1,X2):
    M1 = np.mean(X1)
    M2 = np.mean(X2)
    Res = []
    for i in range(len(X1)):
        val = (X1[i] - M1) * (X2[i] - M2)
        Res.append(val)
    return np.mean(Res)
    
    
def checkSim(Sims,IDs):
    if len(IDs) <= 1:
        return 0
        
    Gets = []
    for i in range(len(IDs)):
        for j in range(i+1,len(IDs)):
            ky1 = IDs[i]
            ky2 = IDs[j]
            Gets.append(Sims[ky1][ky2])
            
    return int(max(Gets) * 1000000) / 1000000
    
    
def cutMean(Vs,C1=75,C2=25):
    cut1 = np.percentile(Vs,C1)
    cut2 = np.percentile(Vs,C2)
    Gets = []
    for l in range(len(Vs)):
        if Vs[l] <= cut1 and Vs[l] >= cut2:
            Gets.append(Vs[l])
    return np.mean(Gets)
    

class MUODV:
    def __init__(self):
        self.AGrades = {}
        self.BGrades = {}
        self.T = 3
        self.Warmup = False
        self.Round = 0
        self.Scores = {}
        self.BNum = []
        
    def MUOD(self,Mat):
        Is = []
        Ia = []
        Im = []
        Sims = []
        for i in range(len(Mat)):
            PCs = []
            Alpha = []
            Beta = []
            sim = []
            for j in range(len(Mat)):
                X = Mat[i]
                Xi = Mat[j]
                pc = pearsonr(X, Xi)[0]
                if np.isnan(pc):
                    pc = 0.00001
                beta = np.cov(X, Xi)[0][1] / max(np.var(Xi),0.00000000001)
                alpha = np.mean(X) - beta * np.mean(Xi)
                if pc < 0.98:
                    PCs.append(pc)
                    Alpha.append(alpha)
                    Beta.append(beta)
                
            Is.append(abs(np.mean(PCs) - 1))
            Im.append(abs(np.mean(Alpha)))
            Ia.append(abs(np.mean(Beta) - 1))

        Loc0 = max(min(LorenzThr(Is),25),10)
        Loc1 = max(min(LorenzThr(Im),25),10)
        Loc2 = max(min(LorenzThr(Ia),25),10)
        Up0 = np.percentile(Is, 100 - Loc0)
        Up1 = np.percentile(Im, 100 - Loc1)
        Up2 = np.percentile(Ia, 100 - Loc2)
        Outlier0 = []
        Outlier1 = []
        Outlier2 = []
        L = len(Mat)
        for i in range(len(Is)):
            val = Is[i]
            if val >= Up0:
                Outlier0.append(i)
            val = Im[i]
            if val >= Up1:
                Outlier1.append(i)
            val = Ia[i]
            if val >= Up2:
                Outlier2.append(i)
        
        Finds = Outlier2
        return Finds
    
    def bestLayer(self):
        Grades = {}
        for ky in self.AGrades.keys():
            grade = self.AGrades[ky] / (self.AGrades[ky] + self.BGrades[ky]) 
            Grades[ky] = grade
    
        Good = list(Grades.keys())
        if self.Warmup:
            Good = []
            SGds = sorted(Grades.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.T):
                Ky = SGds[i][0]
                Good.append(Ky)
        return Good
    
    def detection(self,DMat,UIDs):
        self.Round += 1
        if len(self.AGrades) == 0:
            for i in range(len(DMat)):
                self.AGrades[i] = 1
                self.BGrades[i] = 1
            self.T = max(self.T,int(len(DMat) - 3))
         
        Good = self.bestLayer()
        Votes = {}
        Records = {}
        ThL = 0
        for i in range(len(DMat)):
            Mat = DMat[i]
            layer = i
            Bad = self.MUOD(Mat)
            Records[i] = Bad
            if i in Good:
                if len(Bad) > 0:
                    ThL += 1
                for b in Bad:
                    if b not in Votes.keys():
                        Votes[b] = 1
                    else:
                        Votes[b] += 1
        
        ThL = max(1,ThL)
        Bads = [] 
        Step = 0
        while len(Bads) <= 2:
            Bads = []
            ThL -= Step
            for ky in Votes.keys():
                if Votes[ky] >= ThL:
                    Bads.append(ky)
            Step += 1
            if ThL <= 0:
                break

        L = len(Bads)
        for ky in Records.keys():
            Get = Records[ky]
            VL = len(list(set(Get) & set(Bads)))
            BL = len(Get) - VL
            if VL > L / 2:
                self.AGrades[ky] += 1
            else:
                self.BGrades[ky] += 1
        
        if self.Round >= 5:
            self.Warmup = True
            
        return Bads


class RandomGet:
    def __init__(self, Fixed=False, Nclients=0):
        self.totalArms = OrderedDict()
        self.training_round = 0
        self.IDsPool = []
        self.Fixed = Fixed
        self.Round = 0
        self.Clients = Nclients
        self.FixAIDs = []
        self.ANum = 0
        self.AttNumKeep = True
        self.AttRate = 0

    def updateAtt(self,AttNum,AttRate,Keep):
        self.ANum = AttNum
        self.FixAIDs = []
        self.AttNumKeep = Keep
        self.AttRate = AttRate
        if Keep:
            Num = int(AttRate * self.Clients)
            for i in range(Num):
                self.FixAIDs.append(i)
            

    def register_client(self, clientId):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['status'] = True

    def updateStatus(self,Id,Sta):
        self.totalArms[Id]['status'] = Sta

    def select_participant(self, num_of_clients):
        viable_clients = [x for x in self.totalArms.keys() if self.totalArms[x]['status']]
        return self.getTopK(num_of_clients, viable_clients)

    def getTopK(self, numOfSamples, feasible_clients):
        pickedClients = []
        attackClients = []
        self.Round += 1

        IDs = []
        for i in range(len(feasible_clients)):
            IDs.append(i)
        rd.shuffle(IDs)
        BNum = int(self.AttRate * numOfSamples)
        GNum = numOfSamples - BNum
        
        BIDs = cp.deepcopy(self.FixAIDs)
        rd.shuffle(BIDs)
        for i in range(BNum):
             attackClients.append(BIDs[i])
        
        for i in range(len(IDs)):
            ky = IDs[i]
            if ky not in BIDs:
                pickedClients.append(ky)
            if len(pickedClients) >= GNum:
                break
                
        pickedClients = attackClients + pickedClients
                    
        return pickedClients, attackClients



