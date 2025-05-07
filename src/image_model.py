import cv2
import numpy as np
import pandas as pd

def gray_scale_images(images,model_name):

    if model_name=='CNN':
        stretch_near = cv2.resize(images, (30,30), 
                    interpolation = cv2.INTER_LINEAR)
        
    if model_name=='VGG16':
        stretch_near = cv2.resize(images, (32,32), 
                    interpolation = cv2.INTER_LINEAR)
        
    if model_name=='Xception':
        stretch_near = cv2.resize(images, (80,80), 
                    interpolation = cv2.INTER_LINEAR)

    # Use the cvtColor() function to grayscale the image 
    gray_image = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)/255.0
    
    return gray_image

def extract_data_from_images(mergeset,Images_Rotation,ml_prediction,loaded_model,encoder_json,model_name):
    common_text_detail = []
    for cord_numb in mergeset.values:
        x,y,w,h = cord_numb[0],cord_numb[1],cord_numb[2],cord_numb[3]
        # cv2.rectangle(thresh1,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(Images_Rotation,(x,y),(x+w,y+h),(0,255,0),1)
        
        if model_name=='CNN':
            ml_images        = gray_scale_images(ml_prediction[y:y+h,x:x+w],model_name)  
            resize_ml_images = ml_images.reshape(1,30,30)

        if model_name=='VGG16':
            ml_images1        = gray_scale_images(ml_prediction[y:y+h,x:x+w],model_name)
            # ml_images         = np.stack((ml_images1,) * 3, axis=-1)
            # resize_ml_images  = ml_images.reshape(1,32,32,3)
            ml_images2        = np.array([i+1 if (i>0.12) & (i<0.5) else 0 for i in ml_images1.flatten()]).reshape(32,32)
            ml_images         = np.stack((ml_images2,) * 3, axis=-1)
            resize_ml_images  = ml_images.reshape(1,32,32,3)


        if model_name=='Xception':
            ml_images1        = gray_scale_images(ml_prediction[y:y+h,x:x+w],model_name)
            # ml_images         = np.stack((ml_images1,) * 3, axis=-1)
            # resize_ml_images  = ml_images.reshape(1,32,32,3)
            ml_images2        = np.array([i+1 if (i>0.12) & (i<0.5) else 0 for i in ml_images1.flatten()]).reshape(80,80)
            ml_images         = np.stack((ml_images2,) * 3, axis=-1)
            resize_ml_images  = ml_images.reshape(1,80,80,3)
        
        prediction       = loaded_model.predict(resize_ml_images)
        
        interpretation   = encoder_json.get('{}'.format(np.argmax(prediction,axis=1)[0]))
        
        common_text_detail.append(interpretation)
        # print(x,y,w,h,'-------------------->',interpretation)
        
    return Images_Rotation, ''.join(common_text_detail)

def images_preprocessing(image):
    #image = cv2.imread(r"D:\TrafficAdBar\caption_code\{}.png".format(file_path.split(".")[0]))
    
    # Resizing the Images
    stretch_near = cv2.resize(image, (450, 250),interpolation = cv2.INTER_LINEAR)
    
    # Rotation Operation Perform
    rows,cols,channels= stretch_near.shape 
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-29,1) 
    rotate_30 = cv2.warpAffine(stretch_near,M,(cols,rows)) 
    Images_Rotation = cv2.warpAffine(stretch_near,M,(cols,rows)) 
    ml_prediction   = cv2.warpAffine(stretch_near,M,(cols,rows))
    
    # Convert Image into grayscale
    gray_image = cv2.cvtColor(rotate_30, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray_image,51, 255, cv2.THRESH_OTSU)
    
    # smooth the image to avoid noises
    gray = cv2.medianBlur(thresh1,5)

    thresh       = cv2.adaptiveThreshold(gray,255,1,1,11,1)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    thresh = cv2.dilate(thresh,None,iterations = 1)
    thresh = cv2.erode(thresh,None,iterations = 1)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    common_detail = []
    extra_symbol  = []
    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
        #print(x,y,w,h)
        # if ((h>=100) | ((w>=60) & (w<=180))):
        if (h>=70) &  (w<=180):
            # print(x,y,w,h)
            common_detail.append([x,y,w,h])
            cv2.rectangle(rotate_30,(x,y),(x+w,y+h),(0,255,0),1)
            
        if (h<=30) & (w>=60):
            # print(x,y,w,h)
            extra_symbol.append([x,y,w,h])
    
    dp = pd.DataFrame(extra_symbol,columns=['x','y','w','h'])
    df = pd.DataFrame(common_detail,columns=['x','y','w','h'])
    # print(dp)
    if dp.shape[0]!=0:
        x0  = dp['x'].min()
        y0  = dp['y'].min()-25
        w0  = dp['w'].max()
        h0  = 100
        
    # df = pd.DataFrame(common_detail,columns=['x','y','w','h'])
    relation_index = []
    for idx in range(0,df.shape[0]):
        reference = df.iloc[idx:idx+1,:]
        for jx in range(idx+1,df.shape[0]):
            test_case = df.iloc[jx:jx+1,:]
            examine = reference.values[0]-test_case.values[0]

            trigger = [f for f in examine if abs(f)<=12]
            if len(trigger)==4:
                relation_index.append([idx,jx])
                
    left_sides = df[~df.index.isin(np.array(relation_index).flatten())]
    
    results = []
    for vals in relation_index:
        # print(vals)
        samples = df[df.index.isin(vals)]
        # print(samples.max().values)
        results.append(samples.max().values)
        
    right_sides = pd.DataFrame(results,columns=['x','y','w','h'])
    
    if left_sides.shape[0]!=0:
        tables_ls = left_sides.copy()

    if left_sides.shape[0]==0:
        tables_ls = pd.DataFrame(columns=['x','y','w','h'])

    if right_sides.shape[0]!=0:
        tables_rs = right_sides.copy()

    if right_sides.shape[0]==0:
        tables_rs = pd.DataFrame(columns=['x','y','w','h'])

    mergeset0 = pd.concat([tables_ls,tables_rs])
    mergeset0.index = np.arange(0,mergeset0.shape[0])
    
    mergeset0 = mergeset0.sort_values('x',ascending=True)
    mergeset0.index = np.arange(0,mergeset0.shape[0])
    
    mergeset0['x+w'] = mergeset0['x']+mergeset0['w']
    removing_noise = []
    for idx in mergeset0.index:
        #print(idx)
        sampleset = mergeset0[mergeset0.index==idx]
        x1    = sampleset['x'].values[0]
        width = sampleset['x+w'].values[0]

        referance0 = mergeset0[(mergeset0['x']>=x1) & (mergeset0['x+w']<=width)]
        referance  = referance0[~referance0.index.isin([idx])]

        if referance.shape[0]!=0:
           removing_noise.append(idx) 
        
    mergeset = mergeset0[~mergeset0.index.isin(removing_noise)].iloc[:,:-1] 
    #mergeset0.iloc[:,:-2]#[mergeset0['w+x']>=mergeset0['w+x_']].iloc[:,:-2]
    mergeset = mergeset.sort_values('x',ascending=True)
    mergeset.index = np.arange(0,mergeset.shape[0])
    
    mergeset_info = mergeset.copy()
    if dp.shape[0]!=0:
       # mergeset1 = mergeset.append(pd.DataFrame([[x0,y0,w0,h0]],columns=['x','y','w','h'])) 
        mergeset1 = pd.concat([mergeset,pd.DataFrame([[x0,y0,w0,h0]],columns=['x','y','w','h'])],axis=0)
        mergeset1 = mergeset1.sort_values('x',ascending=True)
        
        mergeset_info  = mergeset1.copy()

    return mergeset_info,Images_Rotation,ml_prediction