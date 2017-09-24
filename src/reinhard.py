from index import *
from transfer import *
import cv2

if __name__ == '__main__':
    originImagePath = '../images/filter/stars_200.jpg'
    targetImagePath = '../images/filter/minotaur_200.jpg'
    
    start = time.clock()
    filterData, filterShape = loadImageData(targetImagePath, True, False)
    originData, originShape = loadImageData(originImagePath, True, False)

    originStd = np.std(originData, axis=0)
    originMeans = np.mean(originData, axis=0)
    targetStd = np.std(filterData, axis=0)
    targetMeans = np.mean(filterData, axis=0)

    img = cv2.imread(originImagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out = np.float_(convert2D(img, img.shape, 3))
   
    out = (out - originMeans) * targetStd / originStd + targetMeans

    # normalization
    outRange = rangeMat(out)
    outRange[outRange > 255] = 255
    outRange[outRange < 0] = 0
    out = normalization(out)
    out = inv_normalization(out, outRange)

    out = data2image(out, img.shape)
    plt.imsave('../Reinhard.png', out)
