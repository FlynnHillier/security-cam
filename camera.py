import cv2
import time
import cv2
from cv2.typing import MatLike
from typing import Tuple

#TODO clean this file up, do what the instructing comments say to do.
#make each 'recorded_frame' input a pandas dataframe with both a frame and a timestamp entry. Append datetime to video when output.
#maybe also record when motion is deteceted.

##credits to https://www.geeksforgeeks.org/webcam-motion-detector-python/ for logic regarding motion detection


Rect = Tuple[int,int,int,int]


def _frames_to_seconds(frame_count:int,fps:int) -> float:
    return frame_count / fps

def _seconds_to_frames(seconds : float | int,fps:int) -> int:
    return int(fps * seconds)




def get_movement_between_frames(
        grayscale_frame_1,
        grayscale_frame_2,
        difference_threshold:int = 30,
        movement_area_threshold: int = 10000,
    ) -> list[Rect]:
        #detect differences between static background and current frame
        difference_frame = cv2.absdiff(grayscale_frame_1,grayscale_frame_2)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(difference_frame, difference_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    
        # Finding contour of moving object
        cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movement_bounding_rects = []

        for contour in cnts:
            if cv2.contourArea(contour) < movement_area_threshold:
                #moving object is not of a valid size? I think - CHECK WHAT THIS DOES
                continue
                        
            movement_bounding_rects.append(cv2.boundingRect(contour))


        return movement_bounding_rects
        








def motion_detect_webcam(
        show_bounding_rects:bool = True,
        prefix_motion_seconds:int = 5,
        suffix_motion_seconds:int = 5,
        show:bool = False,
        refresh_background_rate_seconds:int = 10, # this is the time (in seconds) between which frames are checked to see if a 'new background image' has occured. E.g, motion has stopped but the frame may now look different than it did orignally.
    ):
    cap = cv2.VideoCapture(0)

    ##initialise video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    d_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    d_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"XVID") #avi codec


    ## config
    refresh_background_rate_frames = fps * refresh_background_rate_seconds
    prefix_motion_frames : int = _seconds_to_frames(seconds=prefix_motion_seconds,fps=fps) 
    suffix_motions_frames : int = _seconds_to_frames(seconds=suffix_motion_seconds,fps=fps)

    ## loop vars
    last_motion_detected_epoch : int | None = None
    motion_is_detected : bool = False
    recording : bool = False
    recorded_frames : list[MatLike]= []
    last_sample_frame = None

    static_background = None #for assignment in first loop itteration

    frame_counter = 0

    while cap.isOpened():
        try:
            ret, frame = cap.read()

            if not ret:
                break

            frame_counter += 1

            
            ## detect motion ##
            motion_is_detected = False #default to false
            
            ## convert frame to guassian blur so that changes can be easily detected.
            gs_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gs_frame = cv2.GaussianBlur(gs_frame,(21,21),0)

            if static_background is None:
                static_background = gs_frame #first itteration, set frame as reference (ensure initially still image)
                last_sample_frame = gs_frame
                continue
            
            frame_timestamp = time.time()
            movement_bounding_rects = get_movement_between_frames(static_background,gs_frame)


            if len(movement_bounding_rects) > 0:
                if not motion_is_detected:
                    motion_is_detected = True
                    last_motion_detected_epoch = frame_timestamp
                
                if show_bounding_rects:
                    for (x, y, w, h) in movement_bounding_rects:
                        #draw rectangle around moving object
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


            ## show frame if neccessary ##

            if show:
                cv2.imshow("frame",frame)
                cv2.waitKey(1)


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
                last_sample_frame = gs_frame

            elif recording and not motion_is_detected:
                #motion has ceased being detected. check if last motion detect occured longer ago than the specified motion suffix time.            
                if last_motion_detected_epoch + suffix_motion_seconds < frame_timestamp:
                    recording = False

            elif recording and motion_is_detected:
                #frame has already been appended (default behaviour hence no need to do in if clause)

                if frame_counter % refresh_background_rate_frames == 0:
                    #check to see if the frame is seemingly stationary. e.g it is the same as it was when the last sample was taken.
                    if not last_sample_frame is None:
                        movement_bounding_rects = get_movement_between_frames(last_sample_frame,gs_frame)
                        if len(movement_bounding_rects) == 0:
                            print("frame has seemingly become staionary.")
                            #the frame has likely been stationary since the last sample was taken. (if it is coincedentally in a similar position but motion is still occuring, motion should be redetected within the bounds of the suffix period and recording will resume unbroken)
                            motion_is_detected = False
                            static_background = gs_frame
                    
                    last_sample_frame = gs_frame
                    


            
            if was_recording and not recording:
                #recording has ceased. submit save recording
                print("saving recording.",f"{len(recorded_frames)} frames recorded.")
                
                video_writer = cv2.VideoWriter(filename=f"gen\\{time.time()}.avi",fourcc=fourcc,fps=fps,frameSize=(d_w,d_h))
                for frame in recorded_frames:
                    video_writer.write(frame)

                print("saved recording.")


                #wipe recorded frames to only contain the frames within the specified time to prefix motion.
                recorded_frames = recorded_frames[-prefix_motion_frames:]
        except KeyboardInterrupt:
            break

    video_writer.release()



if __name__ == "__main__":
    motion_detect_webcam(show=True)
    