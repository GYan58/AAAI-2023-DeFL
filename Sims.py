from Settings import *
from Utils import *
from Aggregations import *
from Attacks import *


class Client_Sim:
    def __init__(self, ALoader, PLoader, Model, Lr, wdecay, epoch=1, nclass=10):
        self.TrainData = cp.deepcopy(ALoader)
        self.PTrainData = cp.deepcopy(PLoader)
        self.CXs = None
        self.CYs = None
        self.getCData(ALoader)
        self.DLen = 0 
        self.NClass = nclass
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            self.DLen += len(inputs)
        self.Model = cp.deepcopy(Model)
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Round = 0
        self.LR = Lr
        self.decay_step = 1
        self.decay_rate = 0.9
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        self.vgradnorm = "0,0"

    def reload_data(self, loader):
        self.TrainData = cp.deepcopy(loader)
        
    def getCData(self,Loader):
        XLoader = {}
        YLoader = {}
        self.CXs = {}
        self.CYs = {}
        for i in range(10):
            XLoader[i] = []
            YLoader[i] = []
            
        for batch_id, (inputs, targets) in enumerate(Loader):
            Xs = inputs.cpu().detach().numpy()
            Ys = targets.cpu().detach().numpy()
            for i in range(len(Ys)):
                y = Ys[i]
                x = Xs[i]
                XLoader[y].append(x)
                YLoader[y].append(y)
        
        for ky in range(10):
            X = XLoader[ky]
            Y = YLoader[ky]
            if len(X) >= 5:
                X = np.array(X)
                Y = np.array(Y)
                NX = torch.from_numpy(X).to(device)
                NY = torch.from_numpy(Y).to(device)
                self.CXs[ky] = NX
                self.CYs[ky] = NY
            else:
                self.CXs[ky] = []
                self.CYs[ky] = []

    def getDif(self):
        Res = []
        Ps = []
        for ky in range(10):
            inputs = self.CXs[ky]
            targets = self.CYs[ky]
            Ps.append(len(inputs))
            
            Model = cp.deepcopy(self.Model)
            vgrad_norm = []
            for parms in Model.parameters():
                vgrad_norm.append(0)
            
            if len(inputs) > 1:
                optimizer = torch.optim.SGD(Model.parameters(), lr=self.getLR(), momentum=0.9, weight_decay=self.Wdecay)
                loss_fn = nn.CrossEntropyLoss()
                
                Model.train()
                for e in range(self.Epoch):
                    outputs = Model(inputs)
                    optimizer.zero_grad()
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    c = 0
                    for parms in Model.parameters():
                        gnorm = parms.grad.detach().data.norm(2)
                        vgrad_norm[c] += (gnorm.item()) ** 2
                        c += 1
            
            Res.append(vgrad_norm)
            
        Kys = sorted(Ps,reverse=True)
        
        Have = []  
        Vals = {}
        for i in range(len(Ps)):
            Vals[Ps[i]] = Res[i]
            if Ps[i] != 0:
                Have.append(Res[i])
         
        AvG = list(np.mean(Have,axis=0) * 0.98)
        
        Difs = []
        for ky in Kys:
            Dif = list(Vals[ky]) 
            if ky == 0:
                Dif = AvG
            Difs.append(Dif)
        
        return Difs
    
    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)

    def updateLR(self, lr, drate):
        self.LR = lr
        self.decay_rate = drate

    def getLR(self):
        return self.LR
        
    def getGrad(self,W0,W1):
        Res = cp.deepcopy(W0)
        Lrnow = self.getLR()
        for ky in Res.keys():
            if "weight" in ky or "bias" in ky:
                Res[ky] = (W1[ky] - W0[ky]) / Lrnow / 2
            else:
                Res[ky] -= Res[ky]
        return Res
    
    def getNorm(self,W):
        Kys = W.keys()
        Norm = 0
        Lrnow = self.getLR()
        C = 0
        for ky in Kys:
            if "weight" in ky or "bias" in ky:
                V = W[ky].cpu().detach().numpy()
                Norm += np.linalg.norm(V) ** 2
                C += 1
    
        return Norm * Lrnow

    def selftrain(self,Attack="No"):
        self.Round += 1

        optimizer = torch.optim.SGD(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        if self.Round % self.decay_step == 0:
            self.LR *= self.decay_rate

        TLoader = cp.deepcopy(self.TrainData)
        if Attack == "Part":
            TLoader = cp.deepcopy(self.PTrainData)

        self.difloss = 0
        self.gradnorm = 0
        self.trainloss = 0

        SLoss = []
        GNorm = []
        VGNorm = []
        for parms in self.Model.parameters():
            VGNorm.append(0)

        self.Model.train()
        for r in range(self.Epoch):
            sum_loss = 0
            grad_norm = 0
            C = 0

            for batch_id, (inputs, targets) in enumerate(TLoader):
                C = C + 1
                if Attack == "Label":
                    Ys = self.NClass - targets.detach().numpy() - 1
                    targets = torch.from_numpy(Ys)
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.Model(inputs)
                optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                temp_norm = 0
                c = 0 
                for parms in self.Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                    VGNorm[c] += (gnorm.item()) ** 2
                    c += 1
                
                if grad_norm == 0:
                    grad_norm = temp_norm
                else:
                    grad_norm = grad_norm + temp_norm

                newoutputs = self.Model(inputs)
                newloss = self.loss_fn(newoutputs, targets)
                self.difloss = self.difloss + loss.item() - newloss.item()

            SLoss.append(sum_loss/C)
            GNorm.append(grad_norm)

        self.trainloss = np.mean(SLoss)
        Lrnow = self.getLR()
        self.gradnorm = np.sum(GNorm) * Lrnow
        self.vgradnorm = ""
        for i in range(len(VGNorm)):
            self.vgradnorm += str(VGNorm[i] * Lrnow) + ","
        self.vgradnorm = self.vgradnorm[:-1]
        
    def evaluate(self, loader=None, max_samples=100000):
        self.Model.eval()

        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        if loader == None:
            loader = self.TrainData

        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                iters += 1

                if samples >= max_samples:
                    break

        return correct / samples, loss / iters
    
    def fim(self,loader=None,nout=10):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)

        self.Model.eval()
        Ts = []
        K = 50000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=100, shuffle=False)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=nout,
            variant="classif_logits",
            device="cuda"
        )
        
        Tr = F_Diag.trace().item()

        return Tr


class Server_Sim:
    def __init__(self, TrainLoader, TestLoader, ValidLoader, DifLoader, Model, Lr, wdecay=0, epoch=2, Dname="cifar10"):
        self.TrainData = cp.deepcopy(TrainLoader)
        self.TestData = cp.deepcopy(TestLoader)
        self.ValidData = cp.deepcopy(ValidLoader)
        self.DifData = cp.deepcopy(DifLoader)
        self.Gamma = loadGamma(Dname)
        self.Wdecay = wdecay
        self.Epoch = epoch

        self.Model = cp.deepcopy(Model)
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=self.Gamma)
        self.loss_fn = nn.CrossEntropyLoss()
        self.BackModels = []
        self.RecvParas = []
        self.RecvLens = []
        self.aggAFA = AFA()
        self.aggVGN = VGN()
        
        self.FLD = FLDet()
        self.MUOD = MUODV()
        
        self.KeepParas = []
        self.KeepGParam = None       
        
    def reload_data(self, loader):
        self.TestData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR
    
    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)
        
    def getCosin(self,w0,w1):
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
    
    def minusParas(self, Para1, Multi, Para2):
        Res = cp.deepcopy(Para1)
        for ky in Res.keys():
            Res[ky] = Para1[ky] - Para2[ky] * Multi
        return Res
        
    def getSims(self):
        Sims = []
        RPara = self.getParas()
        GParas = []
        for i in range(len(self.RecvParas)):
            PNow = self.RecvParas[i]
            Update = self.minusParas(PNow,1,RPara)
            GParas.append(Update)
        
        for i in range(len(GParas)):
            X = GParas[i]
            Sim = []
            for j in range(len(GParas)):
                sim = 0
                Sim.append(sim)
            Sims.append(Sim)
        
        for i in range(len(GParas)):
            X = GParas[i]
            Sim = []
            for j in range(i,len(GParas)):
                sim = 0
                if i != j:
                    Xi = GParas[j]
                    sim = self.getCosin(X,Xi)
                    Sims[i][j] = sim
                    Sims[j][i] = sim
        return Sims
        
    def findSims(self,Sims,IDs):
        All = []
        for i in range(len(IDs)):
            K1 = IDs[i]
            for j in range(i,len(IDs)):
                if i != j:
                    K2 = IDs[j]
                    sim = Sims[K1][K2]
                    All.append(sim)
        return All
    
    def addSims(self,Sims,IDs):
        Add = []
        for i in range(len(IDs)):
            K1 = IDs[i]
            for j in range(len(Sims)):
                Sim = Sims[K1][j]
                if Sim >= 0.95 and j not in IDs:
                    Add.append(j)
        Add = list(np.unique(Add))
        return Add
        

    def avgParas(self, Paras, Lens):
        Res = cp.deepcopy(Paras[0])
        Sum = np.sum(Lens)
        for ky in Res.keys():
            Mparas = 0
            for i in range(len(Paras)):
                Pi = Lens[i] / Sum
                Mparas += Paras[i][ky] * Pi
            Res[ky] = Mparas
        return Res


    def aggParas(self, aggmethod, aids, uids, attacknum=0, CLP=0):
        self.KeepGParam = self.getParas()
        self.KeepParas = cp.deepcopy(self.RecvParas)
        UIDs = uids
        frac = attacknum
        if len(self.RecvLens) < 2:
            self.KeepParas = []
            self.RecvParas = []
            self.RecvLens = []
            self.RecvDDFs = []
            self.Model = cp.deepcopy(self.BackModels[-1])
            return 0
        
        GParas = None
        if aggmethod == "FedAvg":
            GParas = self.avgParas(self.RecvParas, self.RecvLens)
        if aggmethod == "MKrum":
            num = max(1, len(self.RecvParas) - frac - 1)
            GParas,_ = AggMKrum(self.RecvParas, frac, num)
        if aggmethod == "TrimMean":
            GParas,_ = AggTrimMean(self.RecvParas, frac)
        if aggmethod == "TrimMed":
            GParas,_ = AggTrimMed(self.RecvParas, frac)
        if aggmethod == "Bulyan":
            num = max(1, len(self.RecvParas) - frac)
            GParas,_ = AggBulyan(self.RecvParas, frac, num)
        if aggmethod == "AFA":
            RPara = self.getParas()
            GParas,_ = self.aggAFA.AggParas(UIDs,RPara,self.RecvParas,self.RecvLens)
        if aggmethod == "FLTrust":
            RPara = self.getParas()
            PLens = []
            PPras = []
            Num = int(len(self.RecvParas) * 0.5)
            for i in range(Num):
                PLens.append(1)
                PPras.append(self.RecvParas[-1-i])
            PurePara = wavgParas(PPras,PLens)
            GParas,_ = AggTrust(PurePara,UIDs,RPara,self.RecvParas,self.RecvLens)
            
            
        if aggmethod == "FLDetector":
            GPara = self.avgParas(self.RecvParas, self.RecvLens)
            RPara = self.getParas()
            Keys = cp.deepcopy(uids)
            MaNum = attacknum
            BeNum = 32 - attacknum
            Bads, TPR, FPR = self.FLD.detection(GPara,self.RecvParas,Keys,RPara,MaNum,BeNum)
            
            if len(Bads) > 0:
                self.Bads = ""
                for b in Bads:
                    self.Bads += str(b) + ","
                self.Bads = self.Bads[:-1]
            
            
            PBads = []
            frac = int(frac / 2)
            MedParas = []
            MedLens = []
            MedIDs = []
            for i in range(len(self.RecvParas)):
                if i not in Bads:
                    MedParas.append(self.RecvParas[i])
                    MedLens.append(self.RecvLens[i])
                    MedIDs.append(uids[i])
                else:
                    PBads.append(uids[i])
            self.RecvParas = cp.deepcopy(MedParas)
            self.RecvLens = cp.deepcopy(MedLens)
            UIDs = MedIDs
            GParas = self.avgParas(self.RecvParas, self.RecvLens)
            
        
        if aggmethod == "DeFL":
            GVecs = []
            RPara = self.getParas()
            for i in range(len(self.RecvParas)):
                gvec = TensorGVec2(self.RecvParas[i],1,RPara)
                GVecs.append(gvec)
                
            DMat = []
            for i in range(len(GVecs[0])):
                Dmat = []
                for j in range(len(GVecs)):
                    Dmat.append(GVecs[j][i])
                DMat.append(Dmat)

            Bads = self.MUOD.detection(DMat,uids)
            RPara = self.getParas()
            GParas,_ = self.aggVGN.AggParas(UIDs,RPara,self.RecvParas,self.RecvLens,Bads,CLP)
                
        self.updateParas(GParas)
        self.KeepParas = []
        self.RecvParas = []
        self.RecvLens = []
        
        self.optimizer.step()
        self.scheduler.step()

        NModel = cp.deepcopy(self.Model)
        self.BackModels.append(NModel)
        if len(self.BackModels) > 2:
            self.BackModels = self.BackModels[1:]
    

    def recvInfo(self, Para, Len):
        self.RecvParas.append(Para)
        self.RecvLens.append(Len)
        

    def evaluate(self, loader=None, max_samples=100000):
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        if loader == None:
            loader = self.TrainData

        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                loss += self.loss_fn(y_, y).item()
                correct += (preds == y).sum().item()
                samples += y_.shape[0]
                iters += 1

                if samples >= max_samples:
                    break

        return loss / iters, correct / samples


    def getVGN(self,Paras,Loader,Epoch=2):
        Model = cp.deepcopy(self.Model)
        Model.load_state_dict(Paras)
        optimizer = torch.optim.SGD(Model.parameters(), lr=self.getLR(), momentum=0.9, weight_decay=self.Wdecay)
        loss_fn = nn.CrossEntropyLoss()
        TLoader = Loader
        
        vgrad_norm = []
        for parms in Model.parameters():
            vgrad_norm.append(0)
        
        Model.train()
        for r in range(Epoch):
            for batch_id, (inputs, targets) in enumerate(TLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = Model(inputs)
                optimizer.zero_grad()
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                c = 0
                for parms in Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    vgrad_norm[c] += (gnorm.item()) ** 2
                    c += 1
        
        Lrnow = self.getLR()

        return np.array(vgrad_norm) * Lrnow

    def evalSingle(self, Loader, Epoch=2):
        if len(self.KeepParas) < 2:
            return [[0, 0], [0, 0]], "0,0 0,0"
        
        VGNs = []
        DWs = "Single "
        for i in range(len(self.KeepParas)):
            ParasNow = self.KeepParas[i]
            VGn = self.getVGN(ParasNow,Loader,Epoch)
            VGNs.append(VGn)
            for v in VGn:
                DWs += str(v) + ","
            DWs = DWs[:-1] + " "
        
        return VGNs, DWs
        
        
    def getDif(self,Paras,Loader,Epoch=1):
        Res = []
    
        for ky in Loader.keys():
            Loader_now = Loader[ky]
            Model = cp.deepcopy(self.Model)
            Model.load_state_dict(Paras)
            optimizer = torch.optim.SGD(Model.parameters(), lr=self.getLR(), momentum=0.9, weight_decay=self.Wdecay)
            loss_fn = nn.CrossEntropyLoss()
            TLoader = Loader_now

            grad_norm = 0
            vgrad_norm = []
            for parms in Model.parameters():
                vgrad_norm.append(0)
                
            Model.train()
            for e in range(Epoch):
                for batch_id, (inputs, targets) in enumerate(TLoader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = Model(inputs)
                    optimizer.zero_grad()
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    c = 0
                    for parms in Model.parameters():
                        gnorm = parms.grad.detach().data.norm(2)
                        vgrad_norm[c] += (gnorm.item()) ** 2
                        c += 1
            
            Res.append(vgrad_norm)
        return Res
    
        
    def evalDif(self, Loader, Epoch=2):
        if len(self.KeepParas) < 2:
            return [[0, 0], [0, 0]], "0,0 0,0"
        Res = []
        DWs = ""
        for i in range(len(self.KeepParas)):
            ParasNow = self.KeepParas[i]
            gDif = self.getDif(ParasNow,Loader,Epoch)
            Res.append(gDif)

            dws = "dif" + str(i+1) + " "
            for j in range(len(gDif)):
                gdif = gDif[j]
                for g in gdif:
                    dws += str(g) + ","
                dws = dws[:-1] + " "
            dws = dws[:-1] + "\n"
            DWs += dws

        return Res, DWs[:-2]






