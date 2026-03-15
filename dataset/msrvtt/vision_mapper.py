from torchvision.transforms.transforms import *
import os
import cv2
import random
import numpy as np
import torch

class VisionMapper(object):
    def __init__(self, d_cfg):
        self.vision = d_cfg['vision']
        self.vision_format = d_cfg['vision_format']
        self.dense_extraction = d_cfg.get('dense_extraction', False)
        self.extract_fps = d_cfg.get('extract_fps', None)
        self.frame_fps = d_cfg.get('frame_fps', None)


        if self.vision_format.startswith('video'):        
            self.sample_num = d_cfg.get('vision_sample_num', None)
        
        self.resolution = 224 # Should be fixed

        self.mean = [0.48145466, 0.4578275, 0.40821073] 
        self.std  = [0.26862954, 0.26130258, 0.27577711]    
        self.vision_transforms =  d_cfg.get('vision_transforms','none')

        if self.vision_transforms == 'crop_flip':
            self.transforms = Compose([
                Resize((self.resolution, self.resolution)),
                Normalize(mean=self.mean, std=self.std),
            ])
        else:
            raise NotImplementedError(f"Are you sure of the current transforms? {self.vision_transforms}")
    
    def read(self, id_):
        if self.vision_format == 'video_rawvideo':
            vision_pixels = []
            sample_num = self.sample_num
            if self.vision_format == 'video_rawvideo': # that's the case for msrvtt
                video_path = os.path.join(self.vision, str(id_))+".mp4"
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < sample_num:
                    cap.release()
                    return None
            
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # List of frame ids
                frames_ids = list(range(total_frames))
                frames_splited = np.array_split(frames_ids, sample_num)
                
                # For now pretrained 
                sample_idx = [i[(len(i) + 1) // 2 - 1] for i in frames_splited]
            
                # Extract
                frames = []
                for idx in sample_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                    ret, frame = cap.read()  # Read the frame
                    if ret:
                        if frame.all()!=None: 
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                            frames.append(frame_rgb)
                
                while len(frames)<sample_num and frames_ids!=[]:
                    idx = random.choice(frames_ids) 
                    frames_ids.remove(idx)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                    ret, frame = cap.read()  # Read the frame
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        frames.append(frame_rgb)

                if len(frames)<sample_num:
                    cap.release()
                    return None
                cap.release()

                frames = np.array(frames) # (8, 480, 848, 3) oppure (8, 1080, 1920, 3) etc.
                
            vision_pixels = torch.from_numpy(frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
            vision_pixels = self.transforms(vision_pixels)
            
            return vision_pixels

        else:
            raise NotImplementedError(f"Vision format {self.vision_format} not implemented yet.")