import os
import glob
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    
    img_paths = glob.glob(os.path.join('img','*.*'))
    for path in img_paths:
        image = Image.open(path)
        basename = os.path.basename(path)
        r_image = frcnn.get_proposal(image)
        r_image.save(os.path.join('proposal_out',basename))
    
    
    
    
    
    
    