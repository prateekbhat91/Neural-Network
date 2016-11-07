import numpy as np


def convert_to_1D(tensor):
    return np.ravel(tensor)

class label_encoder():
    '''
    Encodes the elements of an array from range 0 to num classes -1.
    '''
    def __init__(self):

        self._classes = None
        self.transformation = None

    def fit(self,array):
        '''
        :param array: 1D input numpy array.
        :return: None
        '''
        if array.ndim == 1:
            self._classes, self._transformation = np.unique(array, return_inverse=True)
        else:
            raise ValueError("1D array required")

    def transform(self,array):
        '''
        :param array: 1D numpy array to be transformed
        :return: lebel encoded array.
        '''
        return np.searchsorted(self._classes, array)


    def inverse_transform(self,array):
        '''
        :param array: 1D numpy array to be transformed.
        :return: lable decoded array.
        '''
        if array.ndim == 1:
            return self._classes[array]
        else:
            raise ValueError("1D array required")


# if __name__ == '__main__':
#
#     b = [1,2,4,1,2,5,8,4,5,2,3,4,2]
#     b= np.array(b)
#     le =label_encoder()
#     le.fit(b)
#     b = le.transform(b)
#     # print (b)
#     b = le.inverse_transform(b)
#     print(b)

