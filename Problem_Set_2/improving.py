import os
import sys
import cv2
import win32ui
 
 
 
def imgstitcher(imgs):  # 传入图像数据 列表[] 实现图像拼接
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    _result, pano = stitcher.stitch(imgs)
 
    if _result != cv2.Stitcher_OK:
        print("不能拼接图片, error code = %d" % _result)
        sys.exit(-1)
 
    output = 'result2' + '.png'
    cv2.imwrite(output, pano)
    print("拼接成功. %s 已保存!" % output)
 
 
if __name__ == "__main__":
    # imgPath为图片所在的文件夹相对路径
    imgPath = './Problem2Images/panoramas/grail/'
    
    imgList = os.listdir(imgPath)
    print(imgList)
    imgs = []
    for imgName in imgList[:4]:
        pathImg = os.path.join(imgPath, imgName)
        img = cv2.imread(pathImg)
        if img is None:
            print("图片不能读取：" + imgName)
            sys.exit(-1)
        imgs.append(img)
 
    imgstitcher(imgs)    # 拼接
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()