import numpy as np
from random import *
from matplotlib import  pyplot as plt

# probability model
#class
class probabilityModel:
    randomVariable = []
    #constructor
    
    def __init__(self):  
        randomVariable = np.random.rand(100)
        self.__plotProbs(randomVariable)
    # public
    def generateModel(self):
        print("Generate Model ")
        
    def __plotProbs(self, randomVal):   
        fig, (ax1, ax2) = plt.subplots(1, 2)
        print("Max value ", max(randomVal))
        print("Mean value ", np.mean((randomVal)))
        ax1.set_xlabel('Numbers')
        ax1.set_ylabel('Log value of random number [0,1]')
        ax2.set_xlabel('Numbers')
        ax2.set_ylabel('Value of random number [0,1]')
        ax1.plot(np.log(randomVal))
        ax2.plot(randomVal)
        plt.show()
       
if __name__ == '__main__':
    probm = probabilityModel()
    probm.generateModel()
