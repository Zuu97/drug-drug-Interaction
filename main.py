from model import DDImodel
from variables import*
from util import get_data

if __name__ == "__main__":

    ddi = DDImodel()
    ddi.build_dnn_input()