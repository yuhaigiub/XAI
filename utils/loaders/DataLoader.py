import numpy as np

class DataLoader():
    def __init__(self, xs, ys, ts, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_index = 0
        
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            t_padding = np.repeat(ts[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ts = np.concatenate([ts, t_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ts = ts
    
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, ts = self.xs[permutation], self.ys[permutation], self.ts[permutation]
        self.xs = xs
        self.ys = ys
        self.ts = ts
    
    def get_iterator(self):
        self.current_index = 0

        def _wrapper():
            while self.current_index < self.num_batch:
                start_index = self.batch_size * self.current_index
                end_index = min(self.size, self.batch_size * (self.current_index + 1))
                x_i = self.xs[start_index: end_index, ...]
                y_i = self.ys[start_index: end_index, ...]
                t_i = self.ts[start_index: end_index, ...]
                
                yield (x_i, y_i, t_i)
                self.current_index += 1
        
        return _wrapper()