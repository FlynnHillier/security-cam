import cv2
import time
import cv2
from cv2.typing import MatLike

#TODO clean this file up, do what the instructing comments say to do.

##credits to https://www.geeksforgeeks.org/webcam-motion-detector-python/ for logic regarding motion detection

cap = cv2.VideoCapture(0)

##initialise video writer
fps = cap.get(cv2.CAP_PROP_FPS)
d_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
d_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter.fourcc(*"XVID") #mp4 codec
video_writer = cv2.VideoWriter(filename=r"gen\001.avi",fourcc=fourcc,fps=fps,frameSize=(d_w,d_h))


## config
bounding_rects : bool = True
prefix_motion_frames : int = 72 #change this to a time in future (e.g 5 seconds)
suffix_motion_seconds : int = 5 #how long to keep recording after motion has stopped occuring

## loop vars
last_motion_detected_epoch : int | None = None
motion_is_detected : bool = False
recording : bool = False
recorded_frames : list[MatLike]= [] #maybe always have last ~5 secs of frames recorded at all times so can see prior to motion. Save video async once motion has ended?

static_background = None #for assignment in first loop itteration



while cap.isOpened():
    try:
        ret, frame = cap.read()

        if not ret:
            break

        
        ## detect motion ##
        motion_is_detected = False #default to false
        
        ## convert frame to guassian blur so that changes can be easily detected.
        gs_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gs_frame = cv2.GaussianBlur(gs_frame,(21,21),0)

        if static_background is None:
            static_background = gs_frame #first itteration, set frame as reference (ensure initially still image)
            continue
        
        #detect differences between static background and current frame
        difference_frame = cv2.absdiff(static_background,gs_frame)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(difference_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    
        # Finding contour of moving object
        cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                #moving object is not of a valid size? I think - CHECK WHAT THIS DOES
                continue
            

            if not motion_is_detected:
                last_motion_detected_epoch = int(time.time())
                motion_is_detected = True
            if bounding_rects:
                (x, y, w, h) = cv2.boundingRect(contour)
                # making green rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


        ## record frames if necessary ##
        was_recording = recording
        recorded_frames.append(frame)

        if not recording and not motion_is_detected:
            #delete first frame from recorded frames if more than prefix frames are recorded.
            if len(recorded_frames) > prefix_motion_frames:
                recorded_frames = recorded_frames[1:]

        elif not recording and motion_is_detected:
            print("motion has been detected.")
            #start recording
            recording = True

        elif recording and not motion_is_detected:
            #motion has ceased being detected. check if last motion detect occured longer ago than the specified motion suffix time.            
            if last_motion_detected_epoch + suffix_motion_seconds < time.time(): ## CHANGE THIS LOGIC SO THAT TIME IS SYNCED TO ACTUAL TIME OF FRAME NOT TIME HERE AS IS SUCCEPTABLE TO PROCCESSING LAG.
                recording = False

        elif recording and motion_is_detected:
            #do nothing (frame has already been appended.)
            pass


        
        if was_recording and not recording:
            #recording has ceased. submit save recording
            print("saving recording.",f"{len(recorded_frames)} frames recorded.")
            
            video_writer = cv2.VideoWriter(filename=f"gen\\{time.time()}.avi",fourcc=fourcc,fps=fps,frameSize=(d_w,d_h))
            for frame in recorded_frames:
                video_writer.write(frame)

            print("saved recording.")


            ## INSERT LOGIC TO REVERT RECORDED FRAMES TO POSSESS LAST X MILLISECONDS WORTH OF FRAMES
            recorded_frames = []
    except KeyboardInterrupt:
        break

video_writer.release()




    