import cv2
import time
import os
from multiprocessing import Process, Queue
import numpy as np

# 设置RTSP传输协议为TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPReader:
    """RTSP视频流读取器 - 多进程实现"""
    def __init__(self, rtsp_url, buffer_size=5):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.process = None
    
    def start(self):
        """启动视频读取进程"""
        self.process = Process(target=self.update, args=())
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
        return self.frame_queue.get()
    
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


class RTSPWriter:
    """RTSP视频流写入器 - 多进程实现"""
    def __init__(self, output_url, width, height, fps=30):
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.started = False
        self.stopped = False
        self.frame_queue = Queue(maxsize=10)
        self.process = None
        
    def start(self):
        """初始化RTSP输出流进程"""
        if not self.output_url:
            return self
            
        self.process = Process(target=self.update, args=())
        self.process.daemon = True
        self.process.start()
        self.started = True
        return self
    
    def update(self):
        """处理并写入视频帧的进程"""
        # 配置RTSP输出流，使用H.264编码
        if hasattr(cv2, 'VideoWriter_fourcc'):
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:
            fourcc = cv2.cv.CV_FOURCC(*'H264')
        
        writer = cv2.VideoWriter(self.output_url, fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            print(f"警告: 无法初始化RTSP输出流，尝试使用XVID编码")
            # 尝试使用XVID编码
            if hasattr(cv2, 'VideoWriter_fourcc'):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
            writer = cv2.VideoWriter(self.output_url, fourcc, self.fps, (self.width, self.height))
            
        # 持续写入帧
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is not None:
                    writer.write(frame)
            else:
                time.sleep(0.001)  # 避免CPU空转
                
        # 释放资源
        writer.release()
        
    def write(self, frame):
        """写入帧到队列"""
        if self.started:
            # 如果队列已满，移除最旧的帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            # 添加新帧到队列
            self.frame_queue.put(frame)
            
    def stop(self):
        """停止RTSP写入进程"""
        self.stopped = True
        # 等待进程终止
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=1.0)
            # 如果进程仍在运行，强制结束
            if self.process.is_alive():
                self.process.kill()