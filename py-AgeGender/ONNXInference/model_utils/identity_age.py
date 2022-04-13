import numpy as np

class IdentityAge():
    def __init__(self, l):
        self.l = l

    def identity_age(self):
        if self.l<=9:
            label = self.child(self.l)
        elif self.l>9 and self.l<=21:
            label = self.teenager(self.l)
        elif self.l>21 and self.l<=40:
            label = self.younger(self.l)
        elif self.l>40 and self.l<=71:
            label = self.middle(self.l)
        elif self.l>71:
            label = self.eldely(self.l)
        return label

    def child(self, l):
        if l<=2:
            return 0 #2
        elif l>2 and l<=4:
            return 1 # 4
        elif l>4 and l<=6:
            return 2 # 6
        elif l>6 and l<=9:
            return 3 #9

    def teenager(self, l):
        if l<=12:
            return 4 # 12
        elif l>12 and l<=16:
            return 5 # 16
        elif l>16 and l<=18:
            return 6  # 18
        elif l>18 and l<=21:
            return 7 # 21

    def younger(self, l):
        if l<=24:
            return 8 # 24
        elif l>24 and l<=26:
            return 9 # 26
        elif l>26 and l<=30:
            return 10 # 30
        elif l>30 and l<=35:
            return 11 # 35
        elif l>35 and l<=40:
            return 12 # 40

    def middle(self, l):
        if l<=46:
            return 13 # 46
        elif l>46 and l<=51:
            return 14 # 51
        elif l>51 and l<=58:
            return 15 # 58
        elif l>58 and l<=65:
            return 16 # 65
        elif l>65 and l<=71:
            return 17 # 71

    def eldely(self, l):
        if l<=80:
            return 18 #80
        elif l>80 and l<=90:
            return 19 #90
        elif l>90:
            return 20 # 100
       
def return_generation(age):
    if age <7:
        return 0
    elif age >=7 and age<15:
        return 1
    elif age >=15 and age < 25:
        return 2
    elif age >=25 and age < 45:
        return 3
    elif age >=45 and age <70:
        return 4
    elif age >=70:
        return 5
        

class PostProcess(object):
    def __init__(self, pred_generation, pred_identity):
        self.pred_generation = pred_generation
        self.pred_identity = pred_identity
    
    def generation2identity(self, generation):
        if generation==0:
            return 0, 3
        elif generation==1:
            return 3, 5
        elif generation==2:
            return 6, 9
        elif generation==3:
            return 9, 13
        elif generation==4:
            return 13, 17
        elif generation==5:
            return 17, 20
        
        
    def post_age_process(self):
        lowidx, largeidx = self.generation2identity(np.argmax(self.pred_generation))
        print("lowidx, largeidx from generation", lowidx, largeidx, np.argmax(self.pred_generation))
        if len(self.pred_identity)==1:
            slice_pred_identity = self.pred_identity[0][lowidx:largeidx]
        else:
            slice_pred_identity = self.pred_identity[lowidx:largeidx]
        print('pred_identity', self.pred_identity)
        print('list', slice_pred_identity)
        print("pred identity idx", np.argmax(slice_pred_identity)+lowidx)
        a = np.argmax(slice_pred_identity)+lowidx
        if a==0:
            return 2
        elif a==1:
            return 4
        elif a==2:
            return 6
        elif a==3:
            return 9
        elif a==4:
            return 12
        elif a==5:
            return 16
        elif a==6:
            return 18
        elif a==7:
            return 21
        elif a==8:
            return 24
        elif a==9:
            return 26
        elif a==10:
            return 30
        elif a==11:
            return 35
        elif a==12:
            return 40
        elif a==13:
            return 46
        elif a==14:
            return 51
        elif a==15:
            return 58
        elif a==16:
            return 65
        elif a==17:
            return 71
        elif a==18:
            return 80
        elif a==19:
            return 90
        elif a==20:
            return 100
