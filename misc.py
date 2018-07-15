
import cv2
from torchvision import transforms
from pytvision.transforms import transforms as mtrans


# transformations 
normalize = mtrans.ToMeanNormalization(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    #mean=[0.5, 0.5, 0.5],
    #std=[0.5, 0.5, 0.5]
    )

#normalize = mtrans.ToNormalization()

def get_transforms_aug( size_input ):    
    transforms_aug = transforms.Compose([
        
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        #mtrans.ToResize( (size_input+20, size_input+20), resize_mode='asp' ) ,
        #mtrans.RandomCrop( (size_input, size_input), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ) , 
        
        #------------------------------------------------------------------
        #Geometric 
        
        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE ), 
        mtrans.RandomGeometricalTransform( angle=30, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REPLICATE),
        mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        
        #------------------------------------------------------------------
        #Colors 
        
        mtrans.ToRandomTransform( mtrans.RandomRGBPermutation(), prob=0.30 ),
        mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.15 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.15 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.15 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) ), prob=0.30 ),
        mtrans.ToRandomTransform( mtrans.ToGrayscale(), prob=0.30 ),
                
        #mtrans.ToRandomChoiceTransform( [
        #    mtrans.RandomBrightness( factor=0.15 ), 
        #    mtrans.RandomContrast( factor=0.15 ),
        #    #mtrans.RandomSaturation( factor=0.15 ),
        #    mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) ),
        #    mtrans.RandomGamma( factor=0.30  ),            
        #    mtrans.ToRandomTransform(mtrans.ToGrayscale(), prob=0.15 ),
        #    ]),    
                
        mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.0001), prob=0.25 ),    

        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,
        ])    
    return transforms_aug

def get_transforms_det(size_input):    
    transforms_det = transforms.Compose([
        #mtrans.ToResize( (size_input, size_input), resize_mode='crop' ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        mtrans.ToTensor(),
        normalize,
        ])
    return transforms_det

def get_transforms_hflip(size_input):    
    transforms_hflip = transforms.Compose([
        #mtrans.ToResize( (size_input, size_input), resize_mode='crop' ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        mtrans.HFlip(),
        mtrans.ToTensor(),
        normalize,
        ])
    return transforms_hflip

def get_transforms_gray(size_input):    
    transforms_gray = transforms.Compose([
        #mtrans.ToResize( (size_input, size_input), resize_mode='crop' ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        ])
    return transforms_gray



def get_transforms_aug2( size_input ):    
    transforms_aug = transforms.Compose([
        
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        #mtrans.ToResize( (size_input+20, size_input+20), resize_mode='asp' ) ,
        #mtrans.RandomCrop( (size_input, size_input), limit=0, padding_mode=cv2.BORDER_REFLECT_101  ) , 

        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE ), 
        mtrans.RandomGeometricalTransform( angle=45, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REPLICATE),
        mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        #------------------------------------------------------------------
        #mtrans.RandomRGBPermutation(),
        #mtrans.ToRandomChoiceTransform( [
        #    mtrans.RandomBrightness( factor=0.15 ), 
        #    mtrans.RandomContrast( factor=0.15 ),
        #    #mtrans.RandomSaturation( factor=0.15 ),
        #    mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) ),
        #    mtrans.RandomGamma( factor=0.30  ),            
        #    mtrans.ToRandomTransform(mtrans.ToGrayscale(), prob=0.15 ),
        #    ]),    
        #mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.0001), prob=0.15 ),    

        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,
        ])    
    return transforms_aug

