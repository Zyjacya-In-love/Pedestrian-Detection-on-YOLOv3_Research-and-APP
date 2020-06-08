import cv2


class VideoCamera(object):
    def __init__(self, video_source=0):
        self.video_source = video_source
        # 通过opencv获取实时视频流
        self.camera = cv2.VideoCapture(video_source)
        ## 设置画面的尺寸
        # 画面宽度设定为 1920
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # 画面高度度设定为 1080
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if not self.camera.isOpened():
            raise RuntimeError('Could not start camera.')

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        success, img = self.camera.read()
        if self.video_source == 0:
            ## 图片镜像
            # * 水平翻转 flipCode = 1
            # * 垂直翻转 flipCode = 0
            # * 同时水平翻转与垂直翻转 flipCode = -1
            #
            flipCode = 1
            img = cv2.flip(img, flipCode)
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return img


# Get the number of camera available
def countCameras():
    maxTested = 10
    for i in range(maxTested):
        cap = cv2.VideoCapture(i)
        res = cap.isOpened()
        cap.release()
        if not res:
            return i
    return maxTested