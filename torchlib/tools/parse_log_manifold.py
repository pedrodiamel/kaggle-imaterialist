#!/usr/bin/env python

"""
Parse training log
"""

import os
import sys
import numpy as np 
import pandas as pd
import argparse

# Add path project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def parse_log(filename):
    """Parse log file
    """
    tabletrain = list()
    tableval = list()
    k=0

    with open(filename, 'r') as file:
        for line in file:        
            words = line.split('|')
                               
            newwords = list()
            for w in words: [newwords.append(sw) for sw in w.split()] 

            if len(newwords)==0:
                continue
               
            if newwords[0] == "Trn:": 
                tupla = { 
                    'type':  "Train",
                    'epoch': int(newwords[1]),
                    'iter': int(newwords[2]),
                    'total': int(newwords[3]),
                    'time': float(newwords[5]),
                    'loss': float(newwords[7]),
                    'loss_manifold': float(newwords[9]),
                    'acc': float(newwords[11]),           
                    }
                tabletrain.append(tupla)
                #print('TRAIN: ', newwords)
            
            elif newwords[0] == "Val:": 
                tupla = { 
                    'type':  "Val",
                    'epoch': int(newwords[1]),
                    'iter': int(newwords[2]),
                    'total': int(newwords[3]),
                    'time': float(newwords[5]),
                    'loss': float(newwords[7]),
                    'loss_manifold': float(newwords[9]),
                    'acc': float(newwords[11]),        
                    }
                tableval.append(tupla)                    
                #print('TEST:', tupla)                   
         

    return tabletrain, tableval



def parse_args():
    description = ('Parse a training log into one CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')


    args = parser.parse_args()
    return args


def main():
    
    args = vars( parse_args() )
    print('Load: ', args['logfile_path'] )

    tabletrain, tableval = parse_log( args['logfile_path'] )    
    dftrain = pd.DataFrame( tabletrain )
    dfval = pd.DataFrame( tableval )

    dftrain.to_csv( os.path.join(args['output_dir'], 'log_train.csv') , index=False, encoding='utf-8')
    dfval.to_csv( os.path.join(args['output_dir'], 'log_val.csv') , index=False, encoding='utf-8')

    print('SAVE: ', 'log_train.csv', 'log_val.csv' )
    print('DONE!!!')

    
if __name__ == '__main__':
    main()