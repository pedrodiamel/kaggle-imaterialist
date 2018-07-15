

import sys
import os
import multiprocessing
import csv
import urllib3

from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json
import shutil


from torchlib.datasets.utility import download_images

def parse_data(data_file, out_dir):
    '''
    @data_file
    @return [ key:[id,class], url, output ]
    '''

    ann = {}
    if 'train' in data_file or 'validation' in data_file:
        _ann = json.load(open(data_file))['annotations']
        for a in _ann:
            ann[a['image_id']] = a['label_id']
    packs = []
    j = json.load( open(data_file) )
    images = j['images']
    for item in images:
        assert len(item['url']) == 1
        url = item['url'][0]
        id_ = item['image_id']
        if id_ in ann: id_ = "{}_{}".format(id_, ann[id_])
        packs.append((id_, url, out_dir))
    return packs

def download_imaterialist(data_file, out_dir, workers=3):  
    '''
    @data_file: <train|validation|test.json>
    @out_dir
    '''
  
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    packs = parse_data(data_file,out_dir)
    pool = multiprocessing.Pool( processes=workers )
    with tqdm(total=len(packs)) as t:
        for _ in pool.imap_unordered(download_images, packs ):
            t.update(1)

def create_summition_dummy(data_file):
    pass
    

def refolder( pathimage ):    
    ext = 'jpg'
    files = [ f for f in sorted(os.listdir(pathimage)) if f.split('.')[-1] == ext ];
    for f in files:        
        Id, clss = f.split('.')[0].split('_')
        ifolder = os.path.join( pathimage, clss )
        if not os.path.exists(ifolder):
            print('create folder: ', ifolder)
            os.mkdir(ifolder)

        print('copy {} --> {}'.format(f,'{}.jpg'.format(Id)) )

        ifoldername_src = os.path.join(pathimage, f)
        ifoldername_des = os.path.join(ifolder, '{}.jpg'.format(Id) )      
        shutil.copy(ifoldername_src, ifoldername_des)
        

def main():
    
    if len(sys.argv) != 3:
        print('Syntax: %s <train|validation|test> <output_dir/>' % sys.argv[0])
        sys.exit(0)    
    (data_file, out_dir) = sys.argv[1:]

    try:
       download_imaterialist(data_file, out_dir)
    except:
        pass
    
    if out_dir.split('/')[-1] == 'test':
        return
    refolder( out_dir )

if __name__ == '__main__':
    main()
