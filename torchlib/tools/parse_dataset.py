
#!/usr/bin/env python2
# python parse.py ~/.datasets/test/datasetest/ ~/.datasets/test/datatestnew 64 64 --channels=3 --resize_mode=squash

import argparse
import logging
import os
import random
import requests
import re
import sys
import time
import urllib
import numpy as np
import image
import PIL.Image
from tqdm import tqdm

# Add path project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def make_dirs(dir, extensions):    
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert(False)                
    for dirpath, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(dirpath, fname)
                images.append(path)            
    return images

def parse_test(
    root, 
    dir_output, 
    extensions, 
    image_height=256,
    image_width=256,
    image_channels=3,
    resize_mode='squash',
    ):

    dir_output = os.path.expanduser( dir_output )
    root = os.path.expanduser( root )

    folder_out = os.path.join( dir_output, 'test' )
    folder_in  = os.path.join( root, 'test' )

    images = make_dirs(folder_in, extensions)
    with open( os.path.join(dir_output, '{}.txt'.format( 'test' )) , 'w') as f:
        for i, path in enumerate( tqdm(images) ):
            #print(i, path)              
            image_pathname = os.path.join(folder_out,'{:06d}.jpg'.format(i) )
            # load image
            img = image.load_image(path)
            # transform 
            img = image.resize_image(img,
                                    image_height, image_width,
                                    channels=image_channels,
                                    resize_mode=resize_mode,
                                    )
            # save image
            PIL.Image.fromarray(img).save(  image_pathname )
            # save label
            f.write('{}\n'.format( image_pathname ) )

def parse_train_and_validation(
    root, 
    subfolder, 
    dir_output, 
    extensions, 
    image_height=256,
    image_width=256,
    image_channels=3,
    resize_mode='squash',
    ):
    
    folder_in = os.path.join( root, subfolder )
    folder_out = os.path.join( dir_output, subfolder )

    classes, class_to_idx = find_classes( folder_in )
    samples = make_dataset( folder_in, class_to_idx, extensions )
    if len(samples) == 0:
        raise(RuntimeError("Found 0 files in subfolders of: " + folder_in + "\n"
                            "Supported extensions are: " + ",".join(extensions)))

    with open( os.path.join(dir_output, '{}.txt'.format( 'labels' )) , 'w') as f: 
        for c in classes:
            f.write('{}\n'.format( c ))
            pathname = os.path.join( dir_output, subfolder, c )
            if os.path.exists(pathname) is not True:
                os.makedirs(pathname)

    with open( os.path.join(dir_output, '{}.txt'.format(subfolder)) , 'w') as f: 
        for i, (path, tag) in enumerate( tqdm(samples) ):         
            image_pathname = os.path.join(folder_out, classes[tag] ,'{:06d}.jpg'.format(i) )            
            # load image
            img = image.load_image(path)
            # transform 
            img = image.resize_image(img,
                                    image_height, image_width,
                                    channels=image_channels,
                                    resize_mode=resize_mode,
                                    )
            # save image
            PIL.Image.fromarray(img).save( image_pathname )
            # save label
            f.write('{} {}\n'.format( image_pathname, tag) )
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parse-Folder tool')
    parser.add_argument(
        'dir_input',
        help='A filesystem path to the dataset'
    )    
    parser.add_argument(
        'dir_output',
        help='A filesystem path to the new dataset'
    )
    parser.add_argument('width',
                        type=int,
                        help='width of resized images'
                        )
    parser.add_argument('height',
                        type=int,
                        help='height of resized images'
                        )

    # Optional arguments
    parser.add_argument('-c', '--channels',
                        type=int,
                        default=3,
                        help='channels of resized images (1 for grayscale, 3 for color [default])'
                        )
    parser.add_argument('-r', '--resize_mode',
                        help='resize mode for images (must be "crop", "squash" [default], "fill" or "half_crop")'
                        )
    

    args = vars(parser.parse_args())
    
    dir_input=args['dir_input']
    dir_output=args['dir_output']
    image_height=args['height']
    image_width=args['width']
    image_channels=args['channels']
    resize_mode=args['resize_mode']

    if os.path.exists(dir_output) is not True:
        os.makedirs(dir_output)
        os.makedirs(os.path.join( dir_output, 'train' ))
        os.makedirs(os.path.join( dir_output, 'val' ))
        os.makedirs(os.path.join( dir_output, 'test' ) )

    parse_train_and_validation(dir_input, 'train', dir_output, IMG_EXTENSIONS, image_height, image_width, image_channels, resize_mode)
    parse_train_and_validation(dir_input, 'val', dir_output, IMG_EXTENSIONS, image_height, image_width, image_channels, resize_mode)
    parse_test(dir_input, dir_output, IMG_EXTENSIONS, image_height, image_width, image_channels, resize_mode)


