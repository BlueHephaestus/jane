class FastRunningMean():
    def __init__(self):
        self.x_0 = None #Hidden element that is basically at the beginning of the values array
        self.n = 0
        self.K = 0.0
        self.Ex = 0.0
        self.data = []
        self.values = []
                   
    def pop(self):
        self.n-=1
        x_perc = self.data.pop(0)
        self.Ex -= (x_perc - self.K) 
        
        #Additionally update our 0th element
        self.x_0 = self.values.pop(0)
        
    def push(self, x):
        if self.x_0 != None:
            #Make x the % change if we have our 0th element
            #If we only have the 0th element we use that
            if self.n == 0:
                x_perc = (x/self.x_0)-1
            
            #Otherwise we use our values array
            else:
                x_perc = (x/self.values[-1])-1

            #We append the element
            self.data.append(x_perc)
            self.values.append(x)
            self.n+=1
            self.Ex += x_perc - self.K
        else:
            #Otherwise update our 0th element
            #don't append since we skip this one
            self.x_0 = x
            self.K = 0.0

    def mean(self):
        if self.n == 0:
            return 0
        else:
            return self.K + self.Ex / self.n


