import cv2
import time
from queue import Queue
from threading import Thread
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPReader:
    """RTSP视频流读取器"""
    def __init__(self, rtsp_url, buffer_size=5):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.stopped = False
    
    def start(self):
        """启动视频读取线程"""
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
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
                # 重新应用TCP传输设置
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                cap.set(cv2.CAP_PROP_PROTOCOL_CACHING, 0)
                cap.set(cv2.CAP_PROP_TRANSPORT_RTSP, 0)  # 0表示TCP
                
                if not cap.isOpened():
                    print(f"错误: 重新连接失败")
                    time.sleep(5.0)  # 等待更长时间再尝试
                    continue
                
                continue
                
            # 如果队列已满，移除最旧的帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get()
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
        """停止读取线程"""
        self.stopped = True


class RTSPWriter:
    """RTSP视频流写入器"""
    def __init__(self, output_url, width, height, fps=30):
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.started = False
        
    def start(self):
        """初始化RTSP输出流"""
        if not self.output_url:
            return self
            
        # 配置RTSP输出流，使用H.264编码
        # 注意：这里使用H264编码，如果系统支持的话
        if hasattr(cv2, 'VideoWriter_fourcc'):
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:
            fourcc = cv2.cv.CV_FOURCC(*'H264')
        
        self.writer = cv2.VideoWriter(self.output_url, fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            print(f"警告: 无法初始化RTSP输出流，尝试使用XVID编码")
            # 尝试使用XVID编码
            if hasattr(cv2, 'VideoWriter_fourcc'):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
            self.writer = cv2.VideoWriter(self.output_url, fourcc, self.fps, (self.width, self.height))
        
        self.started = True
        return self
        
    def write(self, frame):
        """写入帧到RTSP流"""
        if self.started and self.writer is not None:
            self.writer.write(frame)
            
    def stop(self):
        """停止RTSP写入"""
        if self.started and self.writer is not None:
            self.writer.release()