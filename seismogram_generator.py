import numpy as np
import torch
import impulse
import matplotlib.pyplot as plt


class Seismogram:
    def __init__(self, shape: tuple, noise: bool = False) -> None:
        self.shape = shape
        self.impulse = impulse.generate_impulse('sin', 30, 30/np.pi/3)
        self.canvas = np.zeros(shape)

    def add_reflection(self, slope: float = 0, padding: int = 0):
        slope_line = np.round((np.sin(np.radians(slope)) * np.linspace(0,128,self.shape[1])),0).astype(np.int32)
        for trace_num in range(self.shape[1]):
            cur_pad = padding + slope_line[trace_num]
            self.canvas[ cur_pad:cur_pad + self.impulse.len,trace_num] = self.impulse.form
        return self
    
    def plot(self) -> None:
        plt.imshow(self.canvas,cmap='gray')
        plt.show()

if __name__ == '__main__':
    seismogram = Seismogram((128, 128))
    seismogram.add_reflection(slope = 30)
    seismogram.plot()