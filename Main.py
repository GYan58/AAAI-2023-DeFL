from Settings import *
from Sims import *
from Utils import *
from Attacks import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class FL_Proc:
    def __init__(self, configs, model):
        self.DataName = configs["dname"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"]
        self.PClients = configs["pclients"] 
        self.IsIID = configs["isIID"]
        self.Alpha = configs["alpha"]
        self.Aug = configs["aug"]  
        self.MaxIter = configs["iters"] 
        self.LogStep = configs["logstep"] 
        self.LR = configs["learning_rate"]
        self.GlobalLR = configs["global_lr"]
        self.WDecay = configs["wdecay"]
        self.BatchSize = configs["batch_size"]
        self.Epoch = configs["epoch"]
        self.DShuffle = configs["data_shuffle"]
        self.Normal = configs["normal"]
        self.FixFedAvg = configs['fix_fedavg'] 
        self.Attack = configs["attack"]
        self.AttackRate = configs["attkrate"]
        self.AttackNumKeep = configs["attkkeep"]
        self.AggMethod = configs["aggmethod"]
        self.Know = configs["know"]
        self.CheckCLP = configs["check_clp"]
        self.ValidP = 0.02
        self.CheckDelta = configs["check_delta"]

        self.Server = None
        self.GModel = model
        self.Clients = {}
        self.ClientLoaders = None
        self.TrainLoader = None
        self.TestLoader = None
        self.ValidLoader = None
        self.DifLoader = None
        self.logpath = None
        self.modpath= None
        self.Pname = ""
        self.updateIDs = []
        self.attackIDs = []
        for i in range(2):
            self.updateIDs.append(i)
        self.TrainRound = 0
        self.Selection = RandomGet(self.FixFedAvg, self.NClients)
        

    def get_train_datas(self):
        self.ClientLoaders, self.TrainLoader, self.TestLoader, self.ValidLoader, self.DifLoader = get_loaders(self.DataName, self.NClients, self.IsIID,self.Alpha, self.Aug, False, False,self.Normal, self.DShuffle,self.BatchSize,self.ValidP)

        
    def logging(self):
        _, teaccu = self.Server.evaluate(self.TestLoader)
        DWs = str(teaccu) + "\n"
        
        with open(self.logpath,"a+") as fw:
            fw.write(DWs) 
        

    def main(self):        
        NClass = {"cifar10":10,"mnist":10,"fmnist":10}
        self.get_train_datas()
        self.Server = Server_Sim(self.TrainLoader, self.TestLoader, self.ValidLoader, self.DifLoader, self.GModel, self.LR, self.WDecay, self.Epoch, self.DataName)
        for c in range(self.NClients):
            self.Clients[c] = Client_Sim(self.ClientLoaders[c], None, self.GModel, self.LR, self.WDecay, self.Epoch, NClass[self.DataName])
            self.Selection.register_client(c)
         
        IDs = []
        for c in range(self.NClients):
            IDs.append(c)
        AttackR = self.AttackRate
        AttackNum = int(AttackR * self.PClients)
        AttackNum = max(0, AttackNum)
        AttackNum = min(AttackNum,len(IDs))
        
        Aph = int(self.Alpha * 100)
        Fname = self.AggMethod + "_" + self.Attack + "_" + str(Aph) + "Aph_" + self.Know
        RPath = BaseRoot + Symbol + "Results/Test" + Symbol
        self.logpath = RPath + Fname + ".log"
        with open(self.logpath,"w") as fw:
            fw.write("") 
         
        self.Selection.updateAtt(AttackNum,AttackR,self.AttackNumKeep)
        ModelCount = 0
        Keys = list(self.Server.getParas().keys())
        Recovered = False 
        NumNaN = 0
        CLPC = CPA(Delta=self.CheckDelta)
        CountNotIn = 0
        RIn = 0
        self.logging()
       
        for It in range(self.MaxIter):            
            updateIDs, attackIDs = self.Selection.select_participant(self.PClients)
            
            print("*"*40)
            print(It + 1, "-th Participants:", updateIDs[-6:])

            GlobalParms = self.Server.getParas()
            TransParas = []
            TransLens = []
            TransGNs = []
            IsNaN = 0
            NaNs = []
            NaNIds = []
            for ky in updateIDs:
                if self.GlobalLR:
                    LrNow = self.Server.getLR()
                    self.Clients[ky].updateLR(LrNow, 1)
                self.Clients[ky].updateParas(GlobalParms)
                if ky in attackIDs:
                    self.Clients[ky].selftrain(self.Attack)
                else:
                    self.Clients[ky].selftrain()

                ParasNow = self.Clients[ky].getParas()
                LenNow = self.Clients[ky].DLen
                GNNow = self.Clients[ky].gradnorm
                
                KId = Keys[-1]
                CheckVec = ParasNow[KId].cpu()
                CheckVal = torch.sum(torch.isnan(CheckVec))
                if CheckVal == 0:
                    TransParas.append(ParasNow)
                    TransLens.append(LenNow)
                    TransGNs.append(GNNow)
                    NaNs.append(0)
                else:
                    IsNaN += 1
                    NaNIds.append(ky)
                    NaNs.append(1)
            
            NattackIDs = []
            NupdateIDs = []
            for l in range(len(attackIDs)):
                if attackIDs[l] not in NaNIds:
                    NattackIDs.append(attackIDs[l])
            for l in range(len(updateIDs)):
                if updateIDs[l] not in NaNIds:
                    NupdateIDs.append(updateIDs[l])
            
            if IsNaN >= len(updateIDs) - 2:
                NumNaN += 1
            if NumNaN >= 10:
                print("NaN Terminate...")
                break
            
            attackIDs = NattackIDs
            updateIDs = NupdateIDs

            if self.CheckCLP:
                RIn = CLPC.Judge(TransLens, TransGNs)
            
            KnowParas = []
            ExtraParas = []
            AttkParas = []
            if self.Know == "Partial":
                VL = max(len(attackIDs),int(len(TransParas) * self.AttackRate * 2))
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        KnowParas.append(TransParas[l])
                
                for l in range(len(TransLens)):
                    if updateIDs[l] not in attackIDs:
                        KnowParas.append(TransParas[l])
                     
                    if len(KnowParas) >= VL:
                        break
                
            if self.Know == "Full":
                KnowParas = cp.deepcopy(TransParas)
                for l in range(len(TransLens)):
                    if updateIDs[l] not in attackIDs:
                        ExtraParas.append(TransParas[l])
                        
            for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        AttkParas.append(TransParas[l])
            
            GenNum = len(attackIDs)            
            if self.Attack == "Fang":
                if self.Know == "Partial":
                    BadParas = attkPKrum(GlobalParms,KnowParas,ExtraParas,GenNum,AttkParas)
                else:
                    BadParas = attkKrum(GlobalParms,KnowParas,ExtraParas,GenNum,AttkParas)
                count = 0
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        TransParas[l] = BadParas[count]
                        count += 1
            
            
            if self.Attack == "MinMax":
                BadParas = attkMinMax(GlobalParms,KnowParas,GenNum,AttkParas)
                count = 0
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        TransParas[l] = BadParas[count]
                        count += 1
                
                
            if self.Attack == "MinSum":
                BadParas = attkMinSum(GlobalParms,KnowParas,GenNum,AttkParas)
                count = 0
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        TransParas[l] = BadParas[count]
                        count += 1
            
            
            if self.Attack == "LIE":
                AllNum = len(updateIDs)
                BadParas = attkLie(GlobalParms,KnowParas,GenNum,AllNum,AttkParas)
                count = 0
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        TransParas[l] = BadParas[count]
                        count += 1

            
            if self.Attack == "No":
                print("* No Attack")

            for l in range(len(TransLens)):
                if NaNs[l] == 0:
                    self.Server.recvInfo(TransParas[l], TransLens[l]) 
            self.Server.aggParas(self.AggMethod,attackIDs,updateIDs,AttackNum,RIn)
            if (It + 1) % self.LogStep == 0:
                self.logging()


if __name__ == '__main__':
    Dataname = "cifar10"
    Type = "alex"
    Model = load_Model(Type, Dataname)

    Num_clients = 128
    Participant = 0.25
    MaxIter = 200
    # AGRs: "DeFL","FLDetector","AFA","FLTrust","MKrum","TrimMean"
    AggMethod = "FLDetector"
    Alpha = 0.5
    # Attacks: Fang, LIE, MinMax, MinSum
    Attack = "MinMax"
    # Number of malicous clients
    AttkRate = 0.125

    Configs = {}
    Configs['pclients'] = int(Participant * Num_clients)
    Configs["alpha"] = Alpha
    Configs["attack"] = Attack
    Configs["aggmethod"] = AggMethod
    Configs["attkrate"] = AttkRate
    Configs["learning_rate"] = 0.01
    Configs["wdecay"] = 1e-5
    Configs["batch_size"] = 16
    Configs["iters"] = MaxIter
    
    Configs["check_clp"] = True
    Configs["check_delta"] = 0.05
    Configs["epoch"] = 2
    Configs["attkkeep"] = True
    Configs['isIID'] = False
    Configs["normal"] = True
    Configs["global_lr"] = True
    Configs["aug"] = False
    Configs["fix_fedavg"] = True
    Configs["data_shuffle"] = True
    Configs['logstep'] = 1
    Configs['dname'] = Dataname
    Configs["mname"] = Type
    Configs['nclients'] = Num_clients

    Knows = ["Full","Partial"]
    for know in Knows:
        Configs["know"] = know
        
        FLSim = FL_Proc(Configs, Model)
        FLSim.main()
        print("* Terminate Training...")
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)



