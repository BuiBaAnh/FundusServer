from PIL import Image
import numpy as np
from predict import predictODCForExp


from PIL import Image
from io import BytesIO
import base64
import zipfile

def experiment(file, phase):
    achive = zipfile.ZipFile(file, 'r')
    allFile = achive.namelist()
    print(allFile)
    if (len(allFile) != 2):
        return str("NumFileError")
    allName = []
    allName.append(allFile[0].split('.')[0])
    allName.append(allFile[1].split('.')[0])
    if (not ('fundus' in allName and 'mask' in allName)):
        return str("NameFileError")
    print(allName)
    if (allFile[0].split('.') == 'fundus'):
        imgFile = allFile[0]
        maskFile = allFile[1]
    else:
        maskFile = allFile[0]
        imgFile = allFile[1]
    imgFile = achive.open(imgFile)
    maskFile = Image.open(achive.open(maskFile))
    mask = np.array(maskFile)
    ret = predictODCForExp(imgFile, phase)
    ret = np.array(ret)
    mask[mask > ret] = 150
    mask[mask < ret] = 50
    ret =Image.fromarray(mask)
    data = BytesIO()
    ret.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data.decode('utf-8')