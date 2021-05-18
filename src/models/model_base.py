from abc import ABC, abstractmethod


class ModelBase(ABC):
    @abstractmethod
    def train(self):
        pass
        
    def test(self):
        pass
        
    def visualize(self):
        pass