import time

import cv2
from PyQt5 import QtCore

from app_controllers.utils.frame_helper import *

'''Thread class for handling the received frames
'''


class WorkerThreadFrame(QtCore.QThread):
    update_camera = QtCore.pyqtSignal(object, object, object, object, object)

    def __init__(self, model, view):
        super(WorkerThreadFrame, self).__init__()
        
        print("=" * 50)
        print("WorkerThreadFrame initialization started")
        
        self.model = model
        self.view = view
        self.inference_model = model.inference_model
        self.slider_brightness = view.slider_brightness
        self.button_rotate = view.button_rotate
        self.slider_contrast = view.slider_contrast
        self.frame = None
        
        # Get camera ID from mapping
        selected_camera_name = view.combobox_camera_list.currentText()
        print(f"Selected camera name: {selected_camera_name}")
        print(f"Camera mapping: {model.camera_mapping}")
        
        self.id = model.camera_mapping.get(selected_camera_name)
        print(f"Camera ID from mapping: {self.id}")
        
        # Fallback to 0 if mapping fails
        if self.id is None:
            print("WARNING: Camera ID is None, defaulting to 0")
            self.id = 0
        
        # IMPORTANT: Close any existing camera connections
        if hasattr(model, 'camera') and model.camera is not None:
            print("Releasing existing camera connection...")
            model.camera.release()
            model.camera = None
        
        print(f"Opening camera at index: {self.id}")
        
        # Try to open the camera without specifying backend first
        self.camera = cv2.VideoCapture(self.id)
        time.sleep(1)  # Give camera time to initialize
        
        if not self.camera.isOpened():
            print("Failed to open camera, trying with explicit backend...")
            self.camera.release()
            self.camera = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
            time.sleep(1)
        
        if not self.camera.isOpened():
            print("DSHOW failed, trying MSMF...")
            self.camera.release()
            self.camera = cv2.VideoCapture(self.id, cv2.CAP_MSMF)
            time.sleep(1)
        
        if not self.camera.isOpened():
            raise Exception(f"Cannot open camera {self.id} - tried all backends")
        
        print("✓ Camera opened successfully!")
        
        # Test read
        ret, test_frame = self.camera.read()
        print(f"Test frame read: {ret}")
        if ret:
            print(f"Test frame shape: {test_frame.shape}")
        else:
            print("WARNING: Camera opened but cannot read frames!")
        
        # Set camera properties - REMOVE MJPG codec
        print("Setting camera properties...")
        # DO NOT use MJPG - it might not be supported
        # self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution set to: {actual_width}x{actual_height}")
        
        self.running = True
        print("WorkerThreadFrame initialization completed")
        print("=" * 50)

    def run(self):
        print("!" * 50)
        print("WorkerThreadFrame.run() STARTED")
        print(f"Camera opened: {self.camera.isOpened()}")
        print(f"Running flag: {self.running}")
        print("!" * 50)
        frame_count = 0
        start_time = time.time()
        fps = 0
        while self.running:
            # read one frame
            b, self.frame = self.camera.read()
            if b:
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
            # change brightness based on slider value
            self.frame = change_brightness(self.frame, self.slider_brightness.value() / 100)
            # change contrast based on slider value
            self.frame = change_contrast(self.frame, self.slider_contrast.value() / 100)
            self.check_orientation()
            self.check_rotation()
            # predict using inference_models
            results = self.inference_model.predict(self.frame)
            self.update_camera.emit(self.model, self.view, self.frame, fps, results)

    def stop(self):
        # terminate the while loop in self.run() method
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()

    def check_rotation(self):
        if self.model.frame_rotation == 90:
            self.frame = np.rot90(self.frame, -1, (0, 1))
        elif self.model.frame_rotation == 180:
            self.frame = np.rot90(self.frame, -2, (0, 1))
        elif self.model.frame_rotation == 270:
            self.frame = np.rot90(self.frame, -3, (0, 1))

    def check_orientation(self):
        if self.model.frame_orientation_vertical == 1:
            self.frame = np.flipud(self.frame)
        if self.model.frame_orientation_horizontal == 1:
            self.frame = np.fliplr(self.frame)
