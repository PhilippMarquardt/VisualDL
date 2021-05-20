from abc import ABC, abstractmethod


class TrainerBase(ABC):
    @abstractmethod
    def train(self):
        pass
