import threading
import time
import json
import queue

import cv2 as cv
import param
from PIL import Image
import panel as pn

from models import process
from models.dlprocess import DeepLearningProcess

pn.extension(design='material', notifications=True, ready_notification='Application fully loaded.')

camera_link = json.load(open("./config/link.json", "r"))

# 使用多线程方法读取摄像头，防止缓存累积
class VideoCapture:
    def __init__(self, name):
        self.cap = cv.VideoCapture(name)
        self.state = False
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.running = True  # Flag to indicate if the thread should keep running
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.state=ret
    
    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        return self.state, self.q.get()

    def stop(self):
        self.running = False
        self.t.join()  # Wait for the thread to exit

class CannotOpenCamera(Exception):
    """Exception raised if the camera cannot be opened."""

class CannotReadCamera(Exception):
    """Exception raised if the camera cannot be read."""


class ServerVideoStream(pn.viewable.Viewer):
    value = param.Parameter(doc="The current snapshot as a Pillow Image")
    paused = param.Boolean(default=False, doc="Whether the video stream is paused")
    trend = pn.indicators.Trend(
        name='FPS', data={'x': [0], 'y': [0]}, width=300, height=200
    )
    count_trend = pn.indicators.Trend(
        name='Count', data={'x': [0], 'y': [0]}, width=300, height=200
    )
    camera_choose = param.ObjectSelector(default="【自推流】高速公路", objects=list(camera_link.keys()))

    # ——————————图像处理模型——————————
    image_process_model = param.ObjectSelector(default="raw", objects=["raw", "sobel", "invert", "roberts", "laplacian"], doc="The model of image processing")
    # ——————————图像处理模型——————————

    # ——————————深度学习图像处理模型——————————
    dl_process_model = param.ObjectSelector(default="none", objects=["none", "face detection", "people counting", "standard segment"], doc="The model of deep learning image processing")
    # ——————————深度学习图像处理模型——————————
    
    # 保存按钮
    save = pn.widgets.Button(name="Save", button_type="primary")

    notifications = pn.state.notifications

    def __init__(self, **params):
        super().__init__(**params)

        self._cameras = {}
        self.camera_index = camera_link[self.camera_choose]
        self.DeepProcess = DeepLearningProcess()

        self._stop_thread = False
        self._thread = threading.Thread(target=self._take_images)
        self._thread.daemon = True
        self.save.param.watch(self._save, "clicks")

    def start(self, camera_indices=None):
        if camera_indices:
            for index in camera_indices:
                self.get_camera(index)

        if not self._thread.is_alive():
            self._thread.start()

    def get_camera(self, index):
        if index in self._cameras:
            return self._cameras[index]

        cap = VideoCapture(index)

        if not cap.isOpened():
            raise CannotOpenCamera(f"Cannot open the camera {index}")
        
        # 如果摄像头个数大于2，则随机删除一个前面的cap，先release，再del
        if len(self._cameras.keys()) > 2:
            old_index = list(self._cameras.keys())[0]
            self._cameras[old_index].stop()
            del self._cameras[old_index]

        self._cameras[index] = cap
        return cap
    
    def notify(self, type, message, duration=2000):
        if self.notifications is None:
            if type == "error":
                self.notifications.error(message, duration=duration)
            elif type == "success":
                self.notifications.success(message, duration=duration)
            elif type == "warning":
                self.notifications.warning(message, duration=duration)
            elif type == "info":
                self.notifications.info(message, duration=duration)
    
    # ——————————图像处理模型——————————
    def process_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.image_process_model == "invert":
            return process.invert(image)
        elif self.image_process_model == "sobel":
            return process.sobel(image)
        elif self.image_process_model == "roberts":
            return process.roberts(image)
        elif self.image_process_model == "laplacian":
            return process.laplacian(image)
        else:
            return image
    # ——————————图像处理模型——————————
    
    # ——————————深度学习图像处理模型——————————
    def dl_process_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.dl_process_model == "face detection":
            return self.DeepProcess.face_detection(image)
        elif self.dl_process_model == "people counting":
            return self.DeepProcess.people_counting(image)
        elif self.dl_process_model == "standard segment":
            return self.DeepProcess.standard_segment(image)
        else:
            return 0, image
    # ——————————深度学习图像处理模型——————————

    @staticmethod
    def _cv2_to_pil(rgb_image):
        image = Image.fromarray(rgb_image)
        return image

    def _take_image(self):
        self.camera_index = camera_link[self.camera_choose]
        camera = self.get_camera(self.camera_index)
        # 降低分辨率，保证传输速度流畅
        ret, frame = camera.read()
        if not ret:
            raise CannotReadCamera("Ensure the camera exists and is not in use by other processes.")
        else:
            count = 0
            if self.dl_process_model == "none":
                frame = self.process_image(frame)
            else:
                count, frame = self.dl_process_image(frame)
            self.count_trend.stream({'x': [self.trend.data['x'][-1] + 1], 'y': [count]}, rollover=50)
            frame = cv.resize(frame, (358, 288), interpolation=cv.INTER_AREA)
            self.value = self._cv2_to_pil(frame)
    
    # 保存
    @param.depends("save.clicks")
    def _save(self, clicks):
        try:
            now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.value.save(f"./save/{now}.jpg")
            pn.state.notifications.error('This is an error notification.')
        except Exception as ex:
            print("Error: Could not save image.")
            print(ex)

    def _take_images(self):
        while not self._stop_thread:
            start_time = time.time()
            if not self.paused:
                try:
                    self._take_image()
                except Exception as ex:
                    print("Error: Could not capture image.")
                    print(ex)
            # 计算当前fps
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            self.trend.stream({'x': [self.trend.data['x'][-1] + 1], 'y': [fps]}, rollover=100)

            # if self.fps > 0:
            #     interval = 1 / self.fps
            #     elapsed_time = time.time() - start_time
            #     sleep_time = max(0, interval - elapsed_time)
            #     time.sleep(sleep_time)

    def __del__(self):
        self._stop_thread = True
        if self._thread.is_alive():
            self._thread.join()
        for camera in self._cameras.values():
            camera.release()
        cv.destroyAllWindows()

    def __panel__(self):
        settings = pn.Column(
            self.param.paused,
            self.trend,
            self.count_trend,
            self.param.camera_choose,
            self.param.image_process_model,
            self.param.dl_process_model,
            self.save,
            width=300,
        )
        image = pn.pane.Image(self.param.value, sizing_mode="stretch_both")
        return pn.Row(settings, image)

server_video_stream = ServerVideoStream()
server_video_stream.start()
server_video_stream.servable()