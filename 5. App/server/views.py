# -*- coding: utf-8 -*-
import cv2
import time
from timeit import default_timer as timer

import yaml
from PIL import Image
from flask import render_template, request, url_for, redirect, flash, Response, jsonify, make_response
from camera_opencv import VideoCamera, countCameras
import numpy as np
from __init__ import app, storage
from util import base64_to_pil, np_to_base64, pil_to_base64, isOpencvImg, Opencv2PIL, PIL2Opencv
from runserver import cfg

YOLO_model = None

if cfg['view']['detection_method'].upper() == 'Darknet'.upper():
    from yolov3_opencv_dnn_detect import YOLO_detector
elif cfg['view']['detection_method'].upper() == 'Keras'.upper():
    from keras_yolov3_detect import YOLO_detector
else :
    raise IOError("Bad Detection method!!!")

YOLO_model = YOLO_detector()

# opencv dnn
def PedestrianDetection(img):
    if cfg['view']['detection_method'].upper() == 'Darknet'.upper():
        if not isOpencvImg(img):
            img = PIL2Opencv(img)
    elif cfg['view']['detection_method'].upper() == 'Keras'.upper():
        if isOpencvImg(img):
            img = Opencv2PIL(img)
    boxes = np.array([])
    if cfg['view'].getboolean('detect_person'):
        boxes, scores = YOLO_model.detect_image(img)
    return boxes



options = [
    {'option_id': '1', 'title': '服务器（远程）摄像头', 'func': 'server_camera'},
    {'option_id': '2', 'title': '客户端（本地）摄像头', 'func': 'client_camera'},
    {'option_id': '3', 'title': '图像（在线预览）', 'func': 'image'},
    {'option_id': '4', 'title': '视频（上传/下载）', 'func': 'video'},
]



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if cfg['view'].getboolean('use_ip_white_list'):
        user_ip = request.remote_addr
        ip_white_list = yaml.load(open('ip_white_list.yml'), Loader=yaml.FullLoader)
        if user_ip not in ip_white_list.values():
            return render_template('errors/unauthorized_ip.html')
    return render_template('index.html', options=options)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/option/<int:option_id>', methods=['GET', 'POST'])
def distribute(option_id):
    return redirect('/' + options[option_id]['func'])



def put_boxes(img, boxes):
    if not isOpencvImg(img):
        img = PIL2Opencv(img)
    # boxes
    if len(boxes) > 0:
        # loop over the indexes we are keeping
        for i in range(len(boxes)):
            # extract the bounding box coordinates
            (x1, y1) = (boxes[i][0], boxes[i][1])
            (x2, y2) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img
def put_FPS_person(img, curr_fps, person_num):
    if not isOpencvImg(img):
        img = PIL2Opencv(img)
    # FPS and person
    text = "FPS: " + str(curr_fps) + "\nperson: " + str(person_num)
    y0, dy = 15, 20
    for i, txt in enumerate(text.split('\n')):
        y = y0 + i * dy
        # cv2.putText(img, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1, 2)
        cv2.putText(img, text=txt, org=(3, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
    # cv2.putText(img, text=text, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.50, color=(255, 0, 0), thickness=2)
    return img
def tes_fps(curr_fps):
    global fps_stack
    fps_stack.append(curr_fps)
    print(fps_stack)
    tmp_stack = [x for x in fps_stack if x < 100]
    print("curr_fps : ", curr_fps)
    if len(tmp_stack) > 1:
        print("mean : ", np.mean(tmp_stack))
        print("max : ", np.max(tmp_stack))
        print("min : ", np.min(tmp_stack))


# 1. 服务器（远程）摄像头
"""--------------------------------------------------------------------------------------------------"""
server_camera_last_time = 0
def gen(camera):
    global server_camera_last_time
    server_camera_last_time = 0
    """Video streaming generator function."""
    while True:
        # start = timer()
        img = camera.get_frame()
        # 调用 Keras 目标检测
        boxes = PedestrianDetection(img)
        # # PIL 转 opencv
        # img = PIL2Opencv(img)
        end = timer()

        seconds = end - server_camera_last_time
        server_camera_last_time = end
        curr_fps = round(1.0 / seconds, 2)
        # img = put_FPS(img, curr_fps)
        # img = put_boxes_FPS_person(img, boxes, curr_fps, len(boxes))
        img = put_boxes(img, boxes)
        img = put_FPS_person(img, curr_fps, len(boxes))
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(VideoCamera(camera_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/server_camera', methods=['GET', 'POST'])
def server_camera():
    CameraNumber = countCameras()
    return render_template('choose_server_camera.html', CameraNumber=CameraNumber)

@app.route('/server_camera_play/<int:camera_id>', methods=['GET', 'POST'])
def server_camera_play(camera_id):
    global server_camera_last_time
    server_camera_last_time = 0
    return render_template('server_camera_video.html', camera_id=camera_id)
"""--------------------------------------------------------------------------------------------------"""

# 2. 客户端（本地）摄像头
"""--------------------------------------------------------------------------------------------------"""
# 三种方法选择
@app.route('/client_camera', methods=['GET', 'POST'])
def client_camera():
    global last_time
    last_time = 0
    return render_template('client_camera_method_choose.html')

init_fps = cfg['client_camera'].getint('init_max_fps')
init_interval_time = 1000 // init_fps
last_time = 0
is_process_next = True

# 1.简单一些，把客户端传到服务器的图片存在某一个位置，然后 html 展示
client_video_feed_path = './static/images/video_feed.jpeg'
@app.route('/webcam_image', methods=['POST'])
def webcam_image():
    global is_process_next
    global last_time
    if request.method == 'POST' and len(request.files) > 0:
        if not is_process_next:
            return Response('Bad')
        is_process_next = False

        file = request.files['image']
        img = Image.open(file)
        if int(request.form['camera_id']) == 0:
            # 左右镜像翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 调用 Keras 目标检测
        boxes = PedestrianDetection(img)
        # # PIL 转 opencv
        # img = PIL2Opencv(img)
        # draw boxes FPS and person
        now_time = timer()
        seconds = now_time - last_time
        last_time = now_time
        curr_fps = round(1.0 / seconds, 2)
        # img = put_FPS(img, curr_fps)
        # img = put_boxes_FPS_person(img, boxes, curr_fps, len(boxes))
        img = put_boxes(img, boxes)
        img = put_FPS_person(img, curr_fps, len(boxes))
        cv2.imwrite(client_video_feed_path, img)
        is_process_next = True

    return Response('Accepted!!!')

@app.route('/client_camera_1', methods=['GET', 'POST'])
def client_camera_1():
    img = Image.new('RGB', (640, 360), (255, 255, 255))
    img.save(client_video_feed_path)
    global is_process_next
    is_process_next = True
    global last_time
    last_time = 0
    return render_template('client_camera_1.html', init_interval_time=init_interval_time)

fps_stack = []
# 方法二、改进自 方法一，图片存下后服务器暂停处理新发来的图片，
# 等客户端图片刷新完成后由客户端发送一个信号 continue_process ，然后服务器再处理新接收的图片
# client_video_feed_path = './static/images/video_feed.jpeg' 与法一存在相同的位置
@app.route('/webcam_image_2', methods=['POST'])
def webcam_image_2():
    global is_process_next
    global last_time
    # print('continue_process' in request.values.keys())
    if 'continue_process' in request.values.keys():
        is_process_next = True
        return Response('continue_process')
    # if request.method == 'POST' and request.form['continue_process'] != None:
    #     print(int(request.form['continue_process']))
    if request.method == 'POST' and len(request.files) > 0:
        if not is_process_next:
            return Response('Bad')
        is_process_next = False

        file = request.files['image']
        img = Image.open(file)
        if int(request.form['camera_id']) == 0:
            # 左右镜像翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 调用 Keras 目标检测
        boxes = PedestrianDetection(img)
        # # PIL 转 opencv
        # img = PIL2Opencv(img)
        # draw boxes FPS and person
        now_time = timer()
        seconds = now_time - last_time
        last_time = now_time
        curr_fps = round(1.0 / seconds, 2)

        global fps_stack
        fps_stack.append(curr_fps)
        print(fps_stack)
        if len(fps_stack) > 1:
            print("curr_fps : ", curr_fps)
            print("mean : ", np.mean(fps_stack[1:]))
            print("max : ", np.max(fps_stack[1:]))
            print("min : ", np.min(fps_stack[1:]))
        # img = put_FPS(img, curr_fps)
        # img = put_boxes_FPS_person(img, boxes, curr_fps, len(boxes))
        img = put_boxes(img, boxes)
        img = put_FPS_person(img, curr_fps, len(boxes))
        cv2.imwrite(client_video_feed_path, img)
        # is_process_next = True
        return Response('refresh')

    return Response('Accepted!!!')

@app.route('/client_camera_2', methods=['GET', 'POST'])
def client_camera_2():
    img = Image.new('RGB', (640, 360), (255, 255, 255))
    img.save(client_video_feed_path)
    global is_process_next
    is_process_next = True
    global last_time
    last_time = 0
    return render_template('client_camera_2.html', init_interval_time=init_interval_time)

## 方法三 全双工通信，客户端图片发过来，处理好，再发回去
@app.route('/webcam_image_3', methods=['POST'])
def webcam_image_3():
    global is_process_next
    global last_time
    if request.method == 'POST':
        if not is_process_next:
            return jsonify(result='')
        # is_process_next = False

        get_data = request.json

        img = base64_to_pil(get_data['file'])

        # print("get_data['camera_id'] : ", get_data['camera_id'])
        if get_data['camera_id'] == 0:
            # 左右镜像翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # YOLO
        boxes = PedestrianDetection(img)
        # img = PIL2Opencv(img)
        # FPS
        now_time = timer()
        seconds = now_time - last_time
        last_time = now_time
        curr_fps = round(1.0 / seconds, 2)

        tes_fps(curr_fps)

        # img = put_FPS(img, curr_fps)
        # img = put_boxes_FPS_person(img, boxes, curr_fps, len(boxes))
        img = put_boxes(img, boxes)
        img = put_FPS_person(img, curr_fps, len(boxes))
        if isOpencvImg(img):
            img = Opencv2PIL(img)
        # img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        result = pil_to_base64(img)
        # Serialize the result, you can add additional fields
        is_process_next = True
        return jsonify(frame_id = get_data['frame_id'], result=result)
    return jsonify(result='')

@app.route('/client_camera_3', methods=['GET', 'POST'])
def client_camera_3():
    global is_process_next
    is_process_next = True
    global last_time
    last_time = 0
    return render_template('client_camera_3.html', init_interval_time=init_interval_time)
"""--------------------------------------------------------------------------------------------------"""


# 3. 图像（在线预览
"""--------------------------------------------------------------------------------------------------"""
@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # 镜像翻转
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = PedestrianDetection(img)
        img = put_boxes(img, boxes)
        if isOpencvImg(img):
            img = Opencv2PIL(img)
        result = pil_to_base64(img)
        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return render_template('image_part.html')
"""--------------------------------------------------------------------------------------------------"""


# 4. 视频（上传/下载）
"""--------------------------------------------------------------------------------------------------"""
num_frames = 0
percent_complete_frame = 0
elap = 0
@app.route('/video', methods=['GET', 'POST'])
def video():
    global percent_complete_frame
    global num_frames
    global elap
    percent_complete_frame = 0
    num_frames = 0
    elap = 0

    if request.method == "POST":

        file = request.files["file"]

        print("File uploaded")
        print(file)

        my_upload = storage.upload(file)

        video_file_name = my_upload.name
        video_file_extension = my_upload.extension
        video_file = app.config.get("STORAGE_CONTAINER") + video_file_name

        vid = cv2.VideoCapture(video_file)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")

        num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        outVideoName = video_file_name.split('.')[0] + "_process." + video_file_extension
        output_path = app.config.get("STORAGE_CONTAINER") + outVideoName
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

        while (vid.isOpened()):
            ret, img = vid.read()
            if ret == True:

                # 计时
                start = time.time()
                # # # 简单尝试处理一下
                # img = cv2.flip(img, 0)
                # 调用 Keras 目标检测
                boxes = PedestrianDetection(img)
                img = put_boxes(img, boxes)
                # PIL 转 opencv
                if not isOpencvImg(img):
                    img = PIL2Opencv(img)
                end = time.time()
                elap = (end - start)
                # write the flipped frame
                out.write(img)
                percent_complete_frame += 1
            else:
                break
        vid.release()
        out.release()

        obj = storage.get(outVideoName)
        view_url = obj.url
        download_url = obj.download_url()
        res = make_response(jsonify({"message": "Process Done", 'view_url': view_url, 'download_url': download_url}), 200)
        return res

    return render_template("upload_video.html")

@app.route("/video_process_task", methods=["GET", "POST"])
def video_process_task():
    global percent_complete_frame
    global num_frames
    global elap
    percent_complete = int((percent_complete_frame/num_frames)*100)
    print(percent_complete_frame, " / ", num_frames)
    print(percent_complete,"%")
    percent_complete = min(percent_complete, 100)
    time_left = round((num_frames-percent_complete_frame)*elap, 2)
    time_left_str = ""
    print(time_left, " s")
    m, s = divmod(time_left, 60)
    h, m = divmod(m, 60)
    # print(s)
    if h == 0:
        if m == 0:
            time_left_str = "%.2f s" % s
        else :
            time_left_str = "%d min %d s" % (m, s)
    else :
        time_left_str = "%d h %d min" % (h, m)
    return jsonify(res=percent_complete, time_left=time_left_str)
"""--------------------------------------------------------------------------------------------------"""
