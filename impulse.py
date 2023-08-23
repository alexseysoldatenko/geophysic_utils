import numpy as np
import torch
import matplotlib.pyplot as plt

class Impulse:
    def __init__(self, len: int = 100,  dt: int = 1,start_phase: bool = False) -> None:
        self.len = len
        self.dt = dt
        self.freq = 1 / self.dt
        self.start_phase = start_phase

    def to_tensor(self, path=''):
        if not hasattr(self, 'form'):
            raise Exception('Form not set')
        
        form_tensor = torch.tensor(self.form)
        torch.save(form_tensor, path)
        
    def get_sin_form(self):
        self.form = np.sin(np.arange(self.len) * self.freq + np.pi*int(self.start_phase))
        return self
    
    def load_from_txt(self, path):
        self.form = np.loadtxt(path)
        return self

    def plot_form(self):
        if not hasattr(self, 'form'):
            raise Exception('Form not set')
        
        plt.plot(self.form)
        plt.show()

def generate_impulse(impulse_type: str, length: int = 100, dt: int = 1, plot: bool = False,start_phase: bool = False) -> Impulse:
    impulse = Impulse(length, dt,start_phase=start_phase)
    
    if impulse_type == 'sin':
        impulse = impulse.get_sin_form()
    
    if plot:
        impulse.plot_form()
    
    return impulse


if __name__ == '__main__':
    generate_impulse('sin', 100, 100/np.pi/2)
    


    