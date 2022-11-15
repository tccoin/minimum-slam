import numpy as np
import cv2
folderPath = "/home/link/Projects/curly_slam/data/tartanair/scenes/seasidetown/Easy/P000"
imageFolder = "/depth_left/"
savePath = folderPath+"/depth_image/"
trajName = "pose_left.txt"
def readimage(imageIndex):
    imageName = '0'*(6 - len(str(imageIndex))) + str(imageIndex) + "_left_depth.npy"
    imageName = folderPath + imageFolder + imageName
    print(imageName)
    depth = np.load(imageName)
    #depth = np.uint16(depth * 1000)
    #print(depth)
    
    cv2.imwrite(savePath + '0'*(6 - len(str(imageIndex))) + str(imageIndex) + "_left_depth.tiff",depth)
    # imageIndex += 1
    # depth8U = np.int8(depth/1000)
    # depth8U = 1.0/50 * depth8U
    # cv2.imshow("test", depth8U)
    # cv2.waitKey(0)
    
if __name__=="__main__":
    i = 0
    while True:
        try:
            readimage(i)
            i += 1
        except:
            break
