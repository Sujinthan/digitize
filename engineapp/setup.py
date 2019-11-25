import os
import cv2
import sys

def changeFolderName ():
    '''
        Move files in "by_class" to "train" folder and change subfolder 
        filename from hex to dec
    '''
    main_src = 'by_class/'
    new_src = 'Images for CNN/train/'
    for filename in os.listdir(main_src):
        src = main_src+filename + "/train_"+filename
        new_location = 'Images\ for\ CNN/train/'
        new_filename = ord(bytes.fromhex(filename).decode('utf-8'))
        cmd = 'mkdir ' + new_location +str(new_filename) 
        os.system(cmd)
    print("done creating folder")
    for filename in os.listdir(main_src):
        src = main_src+filename + "/train_"+filename+"/"
        new_filename = str(ord(bytes.fromhex(filename).decode('utf-8')))
        new_location = 'Images\ for\ CNN/train/'
        cmd = 'mv ' + src + "*" + " " + new_location+new_filename
        print(new_filename)
        os.system(cmd)
        # for img in os.listdir(src):
        #     image = cv2.imread(src+'/' + img)
        #     image = cv2.bitwise_not(image)
        #     #print(cv2.bitwise_not(image))
        #     #new_location = new_src+str(ord(filename.decode("hex")))
        #     try:
        #     #os.stat(new_src+filename)
        #         new_filename = ord(bytes.fromhex(filename).decode("utf-8"))
        #         #print(new_src+str(new_filename))
        #         cv2.imwrite(new_src+str(new_filename) + '/' + img, image)  
        #         #print(new_src + str(new_filename))
        #     except OSError as e:
        #         print ("OS error({0}): {1}").format(e.errno, e.strerror)
    print("done inverting and saving file to Images for CNN directory")

def createTestFolder():
    main_src='by_class/'
    # for filename in os.listdir(main_src):
    #     src = main_src + filename + '/hsf_*/*'
    #     new_location = 'Images\ for\ CNN/test2/'
    #     new_filename = ord(bytes.fromhex(filename).decode("utf-8"))
    #     cmd = 'mkdir ' + new_location+str(new_filename)
    #     os.system(cmd)
    print("done creating folder in test")
    new_src = 'Images\ for\ CNN/train/'
    for filename in os.listdir(main_src):
        src = main_src + filename + '/hsf_*/*'
        new_filename =  str(ord(bytes.fromhex(filename).decode('utf-8')))
        moveTo = new_src +new_filename
        print(new_filename)
        cmd = 'mv '+ src +' '+ moveTo
        print(os.system(cmd))
    print("done moving")
    #mv by_class/39/hsf_*/* Images\ for\ CNN/test/57/
def invertImage():
    main_src = 'Images for CNN/test2/'
    for filename in os.listdir(main_src):
        src = main_src+filename 
        for img in os.listdir(src):
            image = cv2.imread(src+'/' + img)
            image = cv2.bitwise_not(image)
            if(os.stat(src+'/'+img)):
                cv2.imwrite(src+img, image)
            else:
                print("src doesn't exists")
            

def countImages ():
    small = float("inf")
    main_src = 'Images for CNN/train/'
    for filename in os.listdir(main_src):
        count = 0
        src = main_src+filename 
        for img in os.listdir(src):
            count+=1
        if(count < small):
            small = count
    return small
def invertMoverImage ():
    main_src = 'Images for CNN/train/'
    new_src = 'Images for CNN/inverted/'
    temp = 'Images\ for\ CNN/inverted/'
    for filename in os.listdir(main_src):
        cmd = 'mkdir ' + temp +str(filename) 
        os.system(cmd)
        count = 0
        src= main_src+filename
        for img in os.listdir(src):
            if(count < 4133):
                image = cv2.imread(src+'/' + img)
                image = cv2.bitwise_not(image)
                #print(cv2.bitwise_not(image))
                #new_location = new_src+str(ord(filename.decode("hex")))
                try:
                    #os.stat(new_src+filename)
                    #new_filename = ord(bytes.fromhex(filename).decode("utf-8"))
                    
                    cv2.imwrite(new_src+str(filename) + '/' + img, image)  
                    #print(new_src + str(new_filename))
                except OSError as e:
                    print ("OS error({0}): {1}").format(e.errno, e.strerror)
            count+=1

            if(count >= 4133):
                break
    print("done inverting")
if __name__=='__main__':
    changeFolderName()
    #print("starting test folder")
    createTestFolder()
    print(countImages())
    invertImage()
    #invertMoverImage()
