from email.utils import localtime
import cv2 as cv
import numpy as np
import dlib
import time
import csv
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont

# 人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()
# 人脸68个关键点检测器
shape_detector = dlib.shape_predictor('./face_detection/face_recognition/weights/shape_predictor_68_face_landmarks.dat')
# 特征描述符提取器，Resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./face_detection/face_recognition/weights/dlib_face_recognition_resnet_model_v1.dat')


def faceRegister(faceId=1,userName='default',interval=3,faceCount=3):
    cap = cv.VideoCapture(0)

    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    # CSV文件写入
    f = open('./face_detection/data/feature.csv', 'a', newline='')
    csv_writer = csv.writer(f)

    # 采集次数
    collect_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        # 裁剪画幅为0.75倍，提高运行速度
        frame = cv.resize(frame, (int(width*0.75), int(height*0.75)))
        # 镜像
        frame = cv.flip(frame, 1)

        # 检测人脸
        detections = hog_face_detector(frame,1)
        for face in detections:
            x = face.left()
            y = face.top()
            r = face.right()
            b = face.bottom()
            cv.rectangle(frame, (x,y), (r,b), (0,255,0), 3)

            # 关键点检测 关键点类型非列表 需要在后面加.part()才能iterable
            points = shape_detector(frame, face)
            for point in points.parts():
                cv.circle(frame, (point.x, point.y), 2, (0,255,0), -1)
            
            
            if collect_count < faceCount+1:
                now = time.time()
                if now - start_time > interval:
                    # 给予录入者准备时间
                    if collect_count == 0:
                        collect_count += 1
                    else:
                        # 将关键点转化为特征描述符
                        face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)

                        # face_descriptor为dlib类型，转化为列表
                        # print(type(face_descriptor))
                        face_descriptor = [f for f in face_descriptor]

                        # 将人脸id和特征描述符等写入CSV文件
                        csv_writer.writerow([faceId, userName, face_descriptor])

                        start_time = now
                        collect_count += 1
                        print('第'+str(collect_count-1)+'次采集人脸')
            else:
                print('人脸采集完毕')
                return 0

        cv.imshow('Demo', frame)
        if cv.waitKey(10) & 0xff==27:
            break

    f.close()
    cap.release()
    cv.destroyAllWindows()


def getFeatList():
    feature_list = None
    label_list = []
    name_list = []

    with open('./face_detection/data/feature.csv', 'r')  as f:
        csv_reader = csv.reader(f)

        for item in csv_reader:
            faceId = item[0]
            name = item[1]
            face_descriptor = eval(item[2])
            face_descriptor = np.array(face_descriptor, np.float64)
            face_descriptor = np.reshape(face_descriptor, (1,-1))

            label_list.append(faceId)
            name_list.append(name)
            if feature_list is None:
                feature_list = face_descriptor
            else:
                feature_list = np.concatenate((feature_list, face_descriptor), axis=0)
    return feature_list,label_list,name_list


def updateRightInfo(frame,face_info_list,face_img_list):
    left_x = 30
    left_y = 30
    resize_w = 80
    offset_y = 120
    frame_w = frame.shape[1]

    index = 0
    for face in face_info_list[:3]:
        name = face[0]
        time = face[1]
        face_img = face_img_list[index]
        face_img = cv.resize(face_img, (resize_w,resize_w))

        offset_y_value = offset_y * index
        frame[(left_y+offset_y_value):(left_y+offset_y_value+resize_w), -(left_x+resize_w):-left_x] = face_img
        cv.putText(frame, name, ((frame_w-(left_x+resize_w)),(left_y+offset_y_value+resize_w+15)), cv.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv.putText(frame, time, ((frame_w-(left_x+resize_w)),(left_y+offset_y_value+resize_w+30)), cv.FONT_ITALIC, 0.5, (0, 255, 0), 1)

        index += 1

    return frame

def faceRecognize(threashold=0.5, interval=3, resize_w=700, resize_h=400):
    cap = cv.VideoCapture(0)
    # 加载特征
    feature_list,label_list,name_list = getFeatList()
    # 人脸信息字典
    face_time_dict = {}
    # 在画面显示打卡成功时间
    show_time = 0
    fps_time = 0

    face_info_list = []
    face_img_list = []

    f = open('./face_detection/data/attendance.csv', 'a', newline='')
    csv_writer = csv.writer(f)

    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        # 裁剪画幅，提高运行速度
        frame = cv.resize(frame,(resize_w,resize_h))
        # 镜像
        frame = cv.flip(frame, 1)
        # 检测人脸
        detections = hog_face_detector(frame,1)
        for face in detections:
            x = face.left()
            y = face.top()
            r = face.right()
            b = face.bottom()
            cv.rectangle(frame, (x,y), (r,b), (0,255,0), 3)
            # 裁剪出人脸部分
            face_crop = frame[y:b, x:r]

            # 关键点检测 关键点类型非列表 需要在后面加.part()才能iterable
            points = shape_detector(frame, face)
            # 将关键点转化为特征描述符
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)           
            # face_descriptor为dlib类型，转化为列表
            face_descriptor = [f for f in face_descriptor] 
            face_descriptor = np.array(face_descriptor, dtype=np.float64)

            # 计算距离
            distance = np.linalg.norm((face_descriptor-feature_list), axis=1)
            # 最小距离索引
            min_index = np.argmin(distance)
            # 最小距离
            min_distance = distance[min_index]
            print(min_distance)

            if min_distance < threashold:
                predict_id = label_list[min_index]
                predict_name = name_list[min_index]
                cv.putText(frame, predict_name, (x,y-15), cv.FONT_ITALIC, 0.5, (0,255,0), 1)
                # print(predict_id)
                # print(predict_name)           

                need_insert = False
                now = time.time()
                if predict_name in face_time_dict:
                    if (now-face_time_dict[predict_name]) > interval:
                        face_time_dict[predict_name] = now
                        need_insert = True
                    else:
                        need_insert = False
                else:
                    face_time_dict[predict_name] = now
                    need_insert = True

                if need_insert:
                    local_time = time.localtime(face_time_dict[predict_name])
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
                    line = [predict_id,predict_name,min_distance,time_str]
                    csv_writer.writerow(line)
                    show_time = time.time()
                    # print(time_str)

                    face_info_list.insert(0, [predict_name, time_str])
                    face_img_list.insert(0, face_crop) 
         
                if now - show_time < 3:
                    cv.putText(frame, "successfully recorded", (x,b+30), cv.FONT_ITALIC, 0.5, (0,255,0), 1)
        
        fps_now = time.time()            
        fpsText = 1/(fps_now - fps_time)
        fps_time = fps_now
        cv.putText(frame, "FPS:  " + str(round(fpsText,2)), (30, 50), cv.FONT_ITALIC, 0.8, (0, 255, 0), 2)

        if len(face_info_list) > 4:
            face_info_list = face_info_list[0:3]
            face_img_list = face_img_list[0:3]
        frame = updateRightInfo(frame,face_info_list,face_img_list) 

        cv.imshow('Demo', frame)
        if cv.waitKey(10) & 0xff==27:
            break

    f.close()
    cap.release()
    cv.destroyAllWindows()


parser = ArgumentParser()
parser.add_argument('--mode', type=str, default='recog',
                    help='运行模式：reg注册人脸  recog识别人脸')
parser.add_argument('--id', type=int, default=1,
                    help='人脸id')
parser.add_argument('--name', type=str, default='Terry',
                    help='人脸姓名')
parser.add_argument("--interval", type=int, default=5,
                    help="人脸每张间隔时间")                       

args = parser.parse_args()

if __name__ == '__main__':
    mode = args.mode
    if mode == 'reg':
        faceRegister(faceId=args.id, userName=args.name)
    if mode == 'recog':
        faceRecognize(interval=args.interval)



