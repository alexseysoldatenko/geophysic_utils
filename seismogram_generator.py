import numpy as np
import torch
import impulse
import matplotlib.pyplot as plt
import random

from scipy import ndimage

def decision(probability):
    return random.random() < probability

class Seismogram:
    def __init__(self, shape: tuple, noise: bool = False, impulse: impulse.Impulse = None) -> None:
        self.shape = shape
        self.impulse = impulse
        self.canvas = np.zeros(shape)

    def add_reflection(self, slope: float = 0, padding: int = 0, orientation = 'right'):
        slope_line = np.round((np.sin(np.radians(slope)) * np.linspace(0,128,self.shape[1])),0).astype(np.int32)
        for trace_num in range(self.shape[1]):
            cur_pad = padding + slope_line[trace_num]
            if cur_pad < self.shape[0]:
                if orientation == 'right':
                    self.canvas[ cur_pad:cur_pad + self.impulse.len,trace_num] += self.impulse.form[:self.impulse.form.shape[0] - (cur_pad + self.impulse.len - 128)]
                
                elif orientation == 'left':
                    self.canvas[ cur_pad:cur_pad + self.impulse.len,self.shape[1] - 1 - trace_num] += self.impulse.form[:self.impulse.form.shape[0] - (cur_pad + self.impulse.len - 128)]
        return self
    
    def add_hyperbola(self,
                      start_time: int = 0, 
                      start_trace: int = 0, 
                      speed: int = 1, 
                      limits: tuple = (0,128), 
                      blur: bool = False,
                      blur_orientation: str = 'right',
                      blur_factor: float = 0.3):
        shift = - np.sqrt(start_time**2 / (speed**2)).astype(np.int32)
        for trace_num in range(self.shape[1]):
            if trace_num > limits[0] and trace_num < limits[1]:
                time = 2*np.round(np.sqrt((trace_num - start_trace)**2 + start_time**2)/(speed**2),0).astype(np.int32) + 2*shift
                if time < self.shape[0]:
                    if blur:
                        if blur_orientation == 'right':
                            if trace_num > start_trace:
                                self.canvas[time,trace_num] = 1 * (1 - (time-start_time)/self.shape[0]) * blur_factor
                            else:
                                self.canvas[time,trace_num] = 1 * (1 - (time-start_time)/self.shape[0])
                        elif blur_orientation == 'left':
                            if trace_num < start_trace:
                                self.canvas[time,trace_num] = 1 * (1 - (time-start_time)/self.shape[0]) * blur_factor
                            else:
                                self.canvas[time,trace_num] = 1 * (1 - (time-start_time)/self.shape[0])
                    else:
                        self.canvas[time,trace_num] = 1 * (1 - (time-start_time)/self.shape[0])
        return self

    def convolve_with_impulse(self):
        for trace_num in range(self.shape[1]):
            self.canvas[:,trace_num] = np.convolve(self.canvas[:,trace_num], self.impulse.form, mode='same')
        return self
    
    def plot(self) -> None:
        plt.imshow(self.canvas,cmap='gray')
        plt.show()

    def to_tensor(self, path: str =''):
        torch.save(torch.from_numpy(self.canvas.copy()).float(), path)

    def add_normal_noise(self, sigma: float = 0.1):
        self.canvas += np.random.normal(0, sigma, self.canvas.shape)
        return self

    def normalize(self):
        """
        Normalize the canvas by subtracting the minimum value and dividing by the difference between the maximum and minimum values.
        
        Parameters:
            None
        
        Returns:
            self: The normalized canvas.
        """
        self.canvas = (np.max(self.canvas)  - self.canvas) / (np.max(self.canvas) - np.min(self.canvas))
        return self
    
    def median_filter(self):
        """
        Apply a median filter to the canvas image.
        Returns:
            self: The current object instance.
        """
        self.canvas = ndimage.median_filter(self.canvas, size=3)
        return self
    
    def add_artifacts(self):
        pass

if __name__ == '__main__':
    for i in range(30000):
        number_of_hyperbolas = np.random.randint(0,6)
        impulse_len = np.random.randint(3,9)
        number_of_sins = np.random.uniform(1,3)
        phase = bool(random.getrandbits(1))
        impulse_test = impulse.generate_impulse('sin', impulse_len, impulse_len/np.pi/number_of_sins, start_phase=phase)
        seismogram = Seismogram((128, 128), impulse=impulse_test)
        blur_factor = 0.2
        blur_orientation = 'right'
        blur = False
        for _ in range(number_of_hyperbolas):
            speed = np.random.uniform(0.4,0.8)
            start_trace = np.random.randint(10,118)
            start_time = np.random.randint(10 * speed,90 * speed)
            left_limit = np.random.randint(1,3)
            right_limit = np.random.randint(125,128)
            blur = decision(0.2)
            if blur:
                blur_orientation = np.random.choice(['right', 'left'])
                blur_factor = np.random.uniform(0.3,0.8)
            seismogram.add_hyperbola(start_trace = start_trace,speed = speed, start_time=start_time, limits = [left_limit, right_limit], blur = blur, blur_orientation = blur_orientation, blur_factor = blur_factor)
            
        seismogram.convolve_with_impulse()
        
        path = f'test_diff_mask\\one_{i}.pt'
        mask = seismogram.canvas.reshape((1,seismogram.canvas.shape[0],seismogram.canvas.shape[1])).copy()
        right_mask = np.where(mask != 0, 1, 0)
        wrong_mask = np.where(mask != 0, 0, 1)
        print(np.concatenate((right_mask, wrong_mask), axis=0).shape)
        torch.save(torch.from_numpy(np.concatenate((right_mask, wrong_mask), axis=0).copy()).float(), path)
        number_of_reflection = np.random.randint(0,4)
        for refl in range(number_of_reflection):
            impulse_len = np.random.randint(3,40)
            number_of_sins = np.random.uniform(1,6)
            impulse_test = impulse.generate_impulse('sin', impulse_len, impulse_len/np.pi/number_of_sins, start_phase=phase)
            seismogram.impulse = impulse_test
            slope = np.random.randint(0,20)
            padding = np.random.randint(0,80)
            orientation = np.random.choice(['left', 'right'])
            seismogram.add_reflection(slope = slope, padding = padding, orientation = orientation)
        noise_strong = np.random.randint(1,3)
        seismogram.add_normal_noise(sigma=0.1 * noise_strong).normalize().to_tensor(path = f'test_diff\\one_{i}.pt')


    # for i in range(30000):
    #     number_of_reflection = np.random.randint(1,5)
    #     impulse_len = np.random.randint(5,40)
    #     number_of_sins = np.random.uniform(1,5)
    #     impulse_test = impulse.generate_impulse('sin', impulse_len, impulse_len/np.pi/number_of_sins)
    #     seismogram = Seismogram((128, 128), impulse=impulse_test)
    #     for refl in range(number_of_reflection):
    #         impulse_len = np.random.randint(5,40)
    #         number_of_sins = np.random.uniform(1,6)
    #         impulse_test = impulse.generate_impulse('sin', impulse_len, impulse_len/np.pi/number_of_sins)
    #         seismogram.impulse = impulse_test
    #         slope = np.random.randint(0,80)
    #         padding = np.random.randint(0,80)#125 - int(np.sin(np.radians(slope))*127) - impulse_len
    #         orientation = np.random.choice(['left', 'right'])
    #         seismogram.add_reflection(slope = slope, padding = padding, orientation = orientation)
    #     seismogram.add_normal_noise(sigma=0.2).normalize().to_tensor(path = f'test_some\\one_{i}.pt')

        