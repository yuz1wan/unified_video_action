import pyrealsense2 as rs
import numpy as np
# import open3d as o3d
import cv2

class CameraD400(object):
    def __init__(self, camera_id="233622078525"):
        target_serial_number = camera_id  # 替换为你要使用的设备的序列号

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_device(target_serial_number)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(self.device)
        self.mtx = self.getIntrinsics()
    
        self.hole_filling = rs.hole_filling_filter()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        print("cam init ...")
        i = 60
        while i > 0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            i -= 1
        print("cam init done.")

    def get_data(self, hole_filling=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image

    def inpaint(self, img, missing_value=0):
        """
        pip opencv-python == 3.4.8.29
        :param image:
        :param roi: [x0,y0,x1,y1]
        :param missing_value:
        :return:
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        if scale < 1e-3:
            pdb.set_trace()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img

    def getleft(self, obj1):
        index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
        index = np.bitwise_and(obj1[:, 1] < 0.5, index)
        index = np.bitwise_and(obj1[:, 1] > -0.5, index)
        # index = np.bitwise_and(obj1[:, 2] > -0.1, index)
        index = np.bitwise_and(obj1[:, 2] > 0.35, index)
        index = np.bitwise_and(obj1[:, 2] < 0.7, index)
        return obj1[index]

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = (
            color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )
        print("intr", intrinsics)
        mtx = [
            intrinsics.width,
            intrinsics.height,
            intrinsics.ppx,
            intrinsics.ppy,
            intrinsics.fx,
            intrinsics.fy,
        ]
        camIntrinsics = np.array(
            [[mtx[4], 0, mtx[2]], [0, mtx[5], mtx[3]], [0, 0, 1.0]]
        )
        return camIntrinsics

    def __del__(self):
        self.pipeline.stop()

if __name__ == "__main__":
    import time
    import os
    import cv2
    import numpy as np

    output_dir = "./test_real.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cam_agent = CameraD400("233522075695") # TODO: change SN code
    cam_bird = CameraD400("135122074278") # TODO: change SN code
    out = None

    while True:
        if cam_agent is not None:
            img, depth = cam_agent.get_data()
            img = img.copy()
            depth = depth.copy()
            cam_agent_view = img
        else:
            print("Wrong Agent Camera!")
            cam_agent_view = np.zeros((640, 480, 3))
            
        if cam_bird is not None:
            img, depth = cam_bird.get_data()
            img = img.copy()
            depth = depth.copy()
            cam_bird_view = img
        else:
            print("Wrong Bird Camera!")
            cam_bird_view = np.zeros((640, 480, 3))

        if out is None:
            # Initialize the video writer with the shape of the concatenated images
            height, width, _ = cam_bird_view.shape
            video_name = output_dir
            out = cv2.VideoWriter(video_name, fourcc, 30.0, (width * 2, height))

        # Concatenate images horizontally
        concatenated_view = np.concatenate((cam_bird_view, cam_agent_view), axis=1)
        
        # Write the frame to the video
        out.write(concatenated_view)

    if out is not None:
        out.release()
        