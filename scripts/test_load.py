from __future__ import print_function

try:
    import time

    import os
    import sys
    import argparse

    import torch

    from cyphercat.config_utils import Configurator, DataStruct
    from cyphercat.load_data import prep_data
    #, get_lfw, get_tiny_imagenet


except ImportError as e:
    print(e)
    raise ImportError




def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-c", "--config", dest="configfile", default='scripts/configs/config.yml', help="Path to yaml")
    
    #model_parse = parser.add_mutually_exclusive_group()
    #model_parse.add_argument("-r", "--rand_rot_angle", dest="rand_rot_angle", default=0., type=float, help="Random image rotation angle range [deg]")
    #model_parse.add_argument("-f", "--fixed_rot_angle", dest="fixed_rot_angle", nargs=3, type=float, help="(low, high, spacing) fixed image rotation angle [deg]")

    args = parser.parse_args()

    print("Testing")

    print("Python: %s" % sys.version)
    print("Pytorch: %s" % torch.__version__)
    
    # determine device to run network on (runs on gpu if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get configuration file
    configr = Configurator(args.configfile)

    # Directory structures for data and model saving
    data_struct = DataStruct(configr.dataset)
    datapath = data_struct.data_path
    print(datapath)

    prep_data(data_struct)

if __name__ == "__main__":
    main()
