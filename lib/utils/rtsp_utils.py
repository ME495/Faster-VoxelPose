import cv2
import time
import os
from multiprocessing import Process, Queue
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|reorder_queue_size;0|rtsp_flags;prefer_tcp|analyzeduration;0|probesize;32768|sync;ext"


class RTSPReader:
    """RTSP视频流读取器 - 多进程实现"""
    def __init__(self, rtsp_url, buffer_size=5, image_size=None, num_views=4, auto_split=False):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.process = None
        self.image_size = image_size  # 要求的图像尺寸
        self.num_views = num_views    # 视图数量
        self.auto_split = auto_split  # 是否自动拆分多视角画面
    
    def start(self, daemon=True):
        """
        启动视频读取进程
        
        Args:
            daemon: 是否设置为守护进程
        """
        self.process = Process(target=self.update, args=())
        if daemon:
            self.process.daemon = True
        self.process.start()
        return self
    
    def update(self):
        """持续从RTSP流读取帧"""
        # 使用cv2.CAP_FFMPEG后端创建VideoCapture对象
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not cap.isOpened():
            print(f"错误: 无法打开RTSP流，URL: {self.rtsp_url}")
            self.stopped = True
            return
        
        # 设置缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        print(f"成功连接RTSP流: {self.rtsp_url}")
        print(f"视频属性: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}@{cap.get(cv2.CAP_PROP_FPS)}fps")

        # FPS计算变量
        fps = 0
        frame_count = 0
        start_time = time.time()

        while not self.stopped:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法从视频流读取帧，尝试重新连接...")
                time.sleep(0.5)
                
                # 尝试重新连接
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print(f"错误: 重新连接失败")
                    time.sleep(5.0)  # 等待更长时间再尝试
                    continue
                
                continue
            
            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # 每秒更新一次FPS
                fps = frame_count / elapsed_time
                # print(f"RTSPReader FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
            if self.auto_split and self.image_size is None:
                # 自动拆分多视角画面
                views, is_valid = self.split_frame(frame)
                if is_valid:
                    frame = views
                
            # 如果队列已满，移除最旧的帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                    
            # 添加新帧到队列
            self.frame_queue.put(frame)
        
        cap.release()
    
    def read(self):
        """从队列中读取最新帧"""
        if self.frame_queue.empty():
            return None
            
        frame = self.frame_queue.get()
        
        return frame
        
        # 如果不需要自动拆分或者没有指定图像尺寸，则直接返回完整帧
        if not self.auto_split or self.image_size is None:
            return frame
            
        # 自动拆分多视角画面
        views, is_valid = self.split_frame(frame)
        if is_valid:
            return views
        else:
            return frame  # 如果无法拆分，返回原始帧
    
    def read_original(self):
        """读取原始帧，不进行拆分"""
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()
    
    def split_frame(self, frame):
        """
        将一个包含多视角图像的帧拆分为多个单独的视图
        
        Args:
            frame: 输入的完整帧
        
        Returns:
            views: 列表，包含拆分后的各个视图
            is_valid: 布尔值，指示分割是否有效
        """
        if self.image_size is None:
            return None, False
            
        height, width = frame.shape[:2]
        expected_height = self.image_size[1] * 2  # 两行图像
        expected_width = self.image_size[0] * 2   # 两列图像
        
        # 检查整体尺寸是否符合预期
        if height != expected_height or width != expected_width:
            return None, False
        
        # 确定每个视图的尺寸
        if self.num_views == 4:  # 2x2布局
            view_height = height // 2
            view_width = width // 2
            
            views = [
                frame[0:view_height, 0:view_width],                          # 左上
                frame[0:view_height, view_width:width],                      # 右上
                frame[view_height:height, 0:view_width],                     # 左下
                frame[view_height:height, view_width:width]                  # 右下
            ]
            
        else:
            raise ValueError(f"不支持的视图数量: {self.num_views}")
        
        return views, True
    
    def stop(self):
        """停止读取进程"""
        self.stopped = True
        # 等待进程终止
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=1.0)
            # 如果进程仍在运行，强制结束
            if self.process.is_alive():
                self.process.kill()
