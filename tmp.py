import cv2

if __name__=="__main__":
    image_name = "img90.jpg"
    image = cv2.imread(image_name)
    image = cv2.transpose(image)
    cv2.imwrite("hh.jpg", image) 
    
