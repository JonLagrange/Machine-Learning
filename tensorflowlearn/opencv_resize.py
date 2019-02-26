import cv2
pic = cv2.imread('./cv.png')
pic = cv2.resize(pic, (256, 256))  # , interpolation=cv2.INTER_CUBIC
cv2.imshow('', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()