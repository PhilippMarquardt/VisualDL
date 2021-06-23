from abc import ABC, abstractmethod

class ModelBase(ABC):    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
     
    @abstractmethod     
    def visualize(self):
        pass