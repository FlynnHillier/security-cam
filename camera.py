import cv2
import time
from datetime import datetime
from cv2.typing import MatLike
from typing import Tuple,TypedDict
from numpy.typing import NDArray
from threading import Thread,Lock



Rect = Tuple[int,int,int,int]

class Meta_Frame(TypedDict):
    """Args:
        frame: (NDArray)

        timestamp: (float | int) - given as epoch time

        belongs_to: (int) - the id of the video the frame belongs to (for the purpose of writing videos)
    """
    frame:NDArray
    timestamp:float | int
    belongs_to:int



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
        refresh_background_rate_seconds:int = 5, # this is the time (in seconds) between which frames are checked to see if a 'new background image' has occured. E.g, motion has stopped but the frame may now look different than it did orignally.
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
    
    prefice_buffer_frames : list[Meta_Frame] = []
    frames_to_write : list[Meta_Frame] = []

    last_sample_frame = None
    static_background = None #for assignment in first loop itteration

    frame_counter = 0
    current_video_index = 0
    end_video_indexes : list[int] = []

    ## threads

    video_storage_thread = Thread(target=video_store_handler,args=(frames_to_write,end_video_indexes,fps,((d_w,d_h))))
    video_storage_thread.start()


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
            

            meta_frame = {
                "frame":frame,
                "timestamp":frame_timestamp,
                "belongs_to":current_video_index,
            }


            if not recording and not motion_is_detected:
                #delete first frame from recorded frames if more than prefix frames are recorded.
                pass

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
                    frame_counter = 0 #reset because if left running for a long time may cause overflow
                    


            ## pass frames to video store handler
            if recording:
                if not was_recording:
                    #started recording on this frame
                    frames_to_write += prefice_buffer_frames
                frames_to_write.append(meta_frame)
                
            elif was_recording and not recording:
                #stopped recording on this frame
                end_video_indexes.append(current_video_index)
                current_video_index += 1


            ## update prefice buffer frames
            prefice_buffer_frames.append(meta_frame)
            if len(prefice_buffer_frames) > prefix_motion_frames:
                    del prefice_buffer_frames[0]

            print(datetime.now().strftime("%H:%M:%S"))
                
        except KeyboardInterrupt:
            break



#this function is intended to be ran asynchronously and read from 'frames_to_write' which should be passed by reference and appended to in an external thread.
def video_store_handler(frames_to_write:list[Meta_Frame],ended_video_indexes:list[int],fps:int, video_dimensions:Tuple[int,int]):    
    save_delay = 5 # delay in seconds to save video 
    
    ## initialise video writer
    fourcc = cv2.VideoWriter.fourcc(*"XVID") # avi codec. (hard code for now, maybe support other codecs in future)
    vw = cv2.VideoWriter(filename=f"gen\\del_{time.time()}.avi",fourcc=fourcc,fps=fps,frameSize=video_dimensions)

    last_written_to_video_index : int | None = None
    last_frame_write_time : float | None = None


    while True:
        if last_written_to_video_index in ended_video_indexes:
            print("saving the last video. starting a new one.")
            ended_video_indexes.remove(last_written_to_video_index)
            #save video and start a new one
            vw.release()
            vw = cv2.VideoWriter(filename=f"gen\\del_{time.time()}_{last_written_to_video_index + 1}.avi",fourcc=fourcc,fps=fps,frameSize=video_dimensions)

        if len(frames_to_write) == 0:
            #there are no frames that need writing
            continue

        #fetch the next frame
        META_FRAME_TO_WRITE = frames_to_write[0]
        #delete next frame from memory
        del frames_to_write[0]


        ## append timestamp text to frame
        frame_to_write = cv2.putText(META_FRAME_TO_WRITE["frame"],
                                        text=datetime.fromtimestamp(META_FRAME_TO_WRITE["timestamp"]).strftime("%d/%m/%Y - %H:%M:%S %p %Z"),
                                        org=(30,30),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=1.5,
                                        color=(255,0,0),
                                        thickness=2,
                                    )



        vw.write(frame_to_write)
        last_written_to_video_index = META_FRAME_TO_WRITE["belongs_to"]

        #remove the 'belongs_to' part. It is obeselete.










if __name__ == "__main__":
    motion_detect_webcam(show=False)
    