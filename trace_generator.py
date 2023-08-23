import numpy as np
import impulse
import matplotlib.pyplot as plt
import torch

class Trace:
    def __init__(self, number_of_boundaries: int = 1, trace_len : int = 128, impulse: impulse.Impulse = None) -> None:
        self.number_of_boundaries = number_of_boundaries
        self.shape = (trace_len,)
        self.reflection_trace = np.zeros(trace_len)
        self.impulse = impulse

        
    def make_reflection_trace(self):
        """
        Generates the reflection trace for the given object.
        Parameters:
            self (object): The object itself.
        Returns:
            None
        """
        for _ in range(self.number_of_boundaries):
            boundary_index = np.random.randint(0, self.shape[0])
            reflection_value = np.random.uniform(-1, 1)
            self.reflection_trace[boundary_index] = reflection_value
    
    def convolve_with_impulse(self):
        """
        Convolve the reflection trace with the impulse.
        Parameters:
            self (object): The object itself.
        Returns:
            None
        """
        if self.impulse is None:
            raise Exception('Impulse not set')
        self.trace = np.convolve(self.reflection_trace, self.impulse.form,mode='same')

    def plot_trace(self, show_reflection_trace: bool = True, plot_impulse: bool = True):
        """
        Plot the reflection trace.
        Parameters:
            self (object): The object itself.
        Returns:
            None
        """
        if show_reflection_trace:
            plt.plot(self.reflection_trace+1+np.max(self.trace), label='Reflection trace', color='red')
        if plot_impulse:
            plt.plot(self.impulse.form+2+np.max(self.trace)+np.max(self.reflection_trace), label='Impulse', color='blue')
        plt.plot(self.trace, label='Reflection trace', color='black')
        plt.show()
    
    def add_damping(self, type: str='linear'):
        if type == 'linear':
            self.trace *= np.linspace(1,0.1, self.shape[0])
    

    
if __name__ == "__main__":
    impulse_test = impulse.Impulse(len=20, dt=20/np.pi/2).get_unnamed_form()
    trace = Trace(number_of_boundaries=100,impulse=impulse_test, trace_len=1024)
    trace.make_reflection_trace()
    # trace.plot_reflection_trace() 
    trace.convolve_with_impulse()
    trace.add_damping()
    trace.plot_trace()       



    