import cv2
import pandas as pd
import os


def overlaybp(csvfile,videofile,threshold=0.6):
    df = pd.read_csv(csvfile)
    dfcol = df.columns
    
    bp = []

    for i in range(len(dfcol)):
        if '_x' in dfcol[i]:
            bp.append(dfcol[i].split('_x')[0])
        elif '_likelihood' in dfcol[i]:
            pass
        elif 'bodyparts_coords' in dfcol[i]:
            pass
        else:
            bp.append(dfcol[i].split('_y')[0])
    bp = list(set(bp))

    print(bp)

    
    vid = str(videofile)
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height, frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outputFileName = os.path.join(os.path.dirname(vid),'output_' + os.path.basename(vid))
    print(outputFileName)
        
    writer = cv2.VideoWriter(outputFileName, fourcc, fps, (int(cap.get(3)),int(cap.get(4))))
    mySpaceScale, myRadius, myResolution, myFontScale = 60, 12, 1500, 1.5
    maxResDimension = max(width, height)
    
    circleScale, fontScale, spacingScale = int(myRadius / (myResolution / maxResDimension)), float(myFontScale / (myResolution / maxResDimension)), int(mySpaceScale / (myResolution / maxResDimension))
    currRow = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:

            for i in bp:
                currXval,currYval,currprob = df[i+'_x'][currRow],df[i+'_y'][currRow], df[i+'_likelihood'][currRow]
                if float(currprob)>= threshold:
                    cv2.circle(frame, (int(currXval), int(currYval)), circleScale, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                else:
                    pass


            writer.write(frame)
            currRow +=1
    
        else:
            break


    cap.release()
    cv2.destroyAllWindows()
    print('Video generated.')




