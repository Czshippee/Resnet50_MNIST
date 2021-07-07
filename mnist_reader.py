def load_mnist(path):
    import os
    import pandas as pd
    import numpy as np
    """Load MNIST data from 'path'"""

    data = pd.read_csv(path).values
    trainsize = int(0.95*len(data))
    train_data = data[0:trainsize, 1:]  
    train_label = data[0:trainsize, 0]
    test_data = data[trainsize:, 1:]  
    test_label = data[trainsize:, 0]

    return train_data,train_label,test_data,test_label



if __name__ == '__main__':
    file_path = '/data1/scz/data/MNIST/train.csv'
    load_mnist(file_path)
    