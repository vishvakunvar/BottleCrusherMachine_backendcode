import datetime
from flask import Flask, request , Response
import RPi.GPIO as GPIO
from flask_cors import CORS
import threading 
import csv
import json
import os
import sys
import tensorflow as tf
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import json
import cv2
import pandas as pd
import requests

# GPIO setup
config = json.load(open('config.json', 'r'))


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# GPIO setup OUTPUT
GPIO.setup(config["crusher"]["pin"],GPIO.OUT)
GPIO.setup(config["conveyor_fw"]["pin"],GPIO.OUT)
GPIO.setup(config["conveyor_rw"]["pin"],GPIO.OUT)
GPIO.setup(config["printer"]["pin"],GPIO.OUT)
GPIO.setup(config["sorting_bottle"]["pin"],GPIO.OUT)
GPIO.setup(config["sorting_can"]["pin"],GPIO.OUT)

# GPIO setup INPUT

GPIO.setup(config["metal"]["pin"],GPIO.IN)
GPIO.setup(config["polybag"]["pin"],GPIO.IN)
GPIO.setup(config["binful"]["proxysensor"]["pin"],GPIO.IN)






# Global variables
weight = 0
bottleStatus = 0

video_feed_active = False
latestThreadLock = threading.Lock()
weightThreadLock = threading.Lock()
latestFrame = None



def tensorThread():
    global latestFrame
    global latestThreadLock
    try:
        global checkCamera
        global bottleStatus
        global canStatus
        time.sleep(30)
        camera_type = 'picamera'
        IM_WIDTH = 640    #Use smaller resolution for
        IM_HEIGHT = 480   #slightly faster framerate
        isbottle = 0
        # This is needed since the working directory is the object_detection folder.
        sys.path.append('..')

        # Import utilites
        from utils import label_map_util
        from utils import visualization_utils as vis_util

        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
        #MODEL_NAME = 'ssd_inception'
        #MODEL_NAME = 'ssd_mobilenet'


        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'mscoco_label_map.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 90

        ## Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)


        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()
        font = cv2.FONT_HERSHEY_SIMPLEX
        

        # Initialize camera and perform object detection.
        # The camera has to be set up and used differently depending on if it's a
    ###############################################################################################################################
        if camera_type == 'picamera':
            # Initialize Picamera and grab reference to the raw capture
            camera = PiCamera()
            camera.resolution = (IM_WIDTH,IM_HEIGHT)
            camera.framerate = 2
            rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
            rawCapture.truncate(0)
            checkCamera = True

            for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

                t1 = cv2.getTickCount()
            
                # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                frame = np.copy(frame1.array)
                frame.setflags(write=1)
                frame_expanded = np.expand_dims(frame, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                # Draw the results of the detection (aka 'visulaize the results')
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.40)

                cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
                with latestThreadLock:
                    latestFrame = frame.copy()

                # All the results have been drawn on the frame, so it's time to display it.
                #cv2.imshow('Object detector', frame)
                

                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc = 1/time1
                
                try:
                    top_detection_score = scores[0][0]
                    top_detection_num = classes[0][0]
                    top_detection_name = category_index[top_detection_num]['name']
                    if (top_detection_score > 0.4):
                        if top_detection_name == 'bottle':
                            print("bottle")
                            bottleStatus=1
                        elif top_detection_name == 'Tin-can':
                            print("can")
                            canStatus = 1
                        else:           
                            bottleStatus=0
                            canStatus = 0
                            print("Nothing Detected")
                    else:
                        bottleStatus=0
                        canStatus = 0
                        print("Nothing Detected")
                except Exception as e:
                    bottleStatus=0
                    canStatus = 0
                    print(e)
                rawCapture.truncate(0)

            camera.close()
        cv2.destroyAllWindows()
    except:
        print("Camera not connected")


def weightThread():
    global config
    while(True):
        global weight
        EMULATE_HX711=False

        referenceUnit = 1

        if not EMULATE_HX711:
            import RPi.GPIO as GPIO
            from hx711 import HX711
        else:
            from emulated_hx711 import HX711


        hx = HX711(19, 26)

        #hx.set_reference_unit(113)
        referenceValue = config["weight"]["referenceValue"]
        if(referenceValue != 0):
            hx.set_reference_unit(referenceValue)

        hx.reset()

        hx.tare()

        print("Tare done! Add weight now...")


        while True:
            try:

                val = hx.get_weight(5)
                with weightThreadLock:
                    weight = val
                print(weight)
                hx.power_down()
                hx.power_up()
                time.sleep(1)
                config = json.load(open('config.json', 'r'))
                tareValue = config["weight"]["referenceValue"]
                if(tareValue != referenceValue):
                    break
                global bottleStatus
                canStatus= 1
                if(config["metal"]["active"]):
                    GPIO.input(config["metal"]["pin"])
                if((weight<-5 and weight>70)and (bottleStatus==0 and canStatus==1) ):
                    break
                
            except (KeyboardInterrupt, SystemExit):
                print("Cleaning...")

                if not EMULATE_HX711:
                    GPIO.cleanup()

                print("Bye!")
                sys.exit()


app = Flask(__name__)
CORS(app)

@app.route('/get-machine-info', methods=['GET'])
def get_machine_info():
    config = json.load(open('config.json', 'r'))
    return json.dumps(config["machineInfo"])

@app.route('/update-machine-info', methods=['POST'])
def set_machine_info():
    data = request.get_json()
    config["machineInfo"] = data
    json.dump(config, open('config.json','w'), indent=4)
    return json.dumps(config["machineInfo"])

@app.route('/get-metal-status', methods=['GET'])
def get_metal_status():
    return json.dumps(config["metal"])

@app.route('/bypass-metal', methods=['POST'])
def bypass_metal():
    data = request.get_json()
    config["metal"]["active"] = data["active"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["metal"])

@app.route('/get-metal-data', methods=['GET'])
def get_metal_data():
    data = GPIO.input(config["metal"]["pin"])
    #data = 0
    data = not data
    return json.dumps({"metal": data})

@app.route('/get-polybag-status', methods=['GET'])
def get_polybag_status():
    return json.dumps(config["polybag"])

@app.route('/bypass-polybag', methods=['POST'])
def bypass_polybag():
    data = request.get_json()
    config["polybag"]["active"] = data["active"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["polybag"])

@app.route('/get-polybag-data', methods=['GET'])
def get_polybag_data():
    data = GPIO.input(config["polybag"]["pin"])
    # data = 0
    data = not data
    return json.dumps({"polybag": data})

@app.route('/get-binfull-status')
def get_binfull_status():
    return json.dumps(config["binful"])

@app.route('/bypass-binfull', methods=['POST'])
def bypass_binfull():
   
    data = request.get_json()
    config["binful"]["active"] = data["active"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["binful"])

@app.route('/get-binfull-data')
def get_binfull_data():
    data = GPIO.input(config["binful"]["proxysensor"]["pin"])
    # data = 0
    data = not data
    return json.dumps({"binful": data})


@app.route('/sesor-data', methods=['GET'])
def get_sensor_data():
    return "{}"

@app.route('/get-password', methods=['GET'])
def get_password():
    return json.dumps(config["password"])

@app.route('/set-password', methods=['POST'])
def set_password():
    data = request.get_json()
    config["password"]["admin"] = data["password"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["password"])

@app.route('/get-crusher-conv-info', methods=['GET'])
def get_crusher_conv_info():
    data = {
        "crusher": config["crusher"],
        "conveyor_fw": config["conveyor_fw"],
        "conveyor_rw": config["conveyor_rw"]
    }

    return json.dumps(data)

@app.route('/gpio-trigger', methods=['POST'])
def gpio_trigger():
    data = request.get_json()
    print(data["pin"])
    print(data["trig"])
    GPIO.output(data["pin"],data["trig"])
    return json.dumps({"status": "success"})

@app.route('/get-machine-data', methods=['GET'])
def get_machine_data():
    data = config["data"]
    return json.dumps(data)

@app.route('/clear-machine-data', methods=['GET'])
def clear_machine_data():
    config["data"] = {
        "bottles": 0,
        "cans": 0,
        "polybags": 0,
    }
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["data"])

@app.route('/get-weight-status', methods=['GET'])
def get_weight_status():
    return json.dumps(config["weight"])

@app.route('/bypass-weight', methods=['POST'])
def bypass_weight():
    data = request.get_json()
    config["weight"]["active"] = data["active"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps(config["weight"])

@app.route('/get-weight-data', methods=['GET'])
def get_weight_data():
    return json.dumps({"weight": weight})

@app.route('/get-flap-status', methods=['GET'])
def get_flap_status():
    data =  {
        "sorting_bottle": config["sorting_bottle"],
        "sorting_can": config["sorting_can"]
    }
    return json.dumps(data)


def generate_frames():
    global video_feed_active
    global latestFrame
    
    while video_feed_active:
        # Read a frame from the webcam
        frame = None
        with latestThreadLock:
            frame = latestFrame.copy()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert the image buffer to bytes
        frame_bytes = buffer.tobytes()

        # Yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the webcam and resources
    cap.release()

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video_feed', methods=['GET'])
def start_video_feed():
    global video_feed_active
    video_feed_active = True
    return json.dumps({"status": "success"})

@app.route('/stop_video_feed', methods=['GET'])
def stop_video_feed():
    global video_feed_active
    video_feed_active = False
    return json.dumps({"status": "success"})

@app.route('/get-all-sensor-data', methods=['GET'])
def get_all_sensor_data():
    global weight
    global bottleStatus

    tempweight = weight
    polybag = GPIO.input(config["polybag"]["pin"])
    metal = GPIO.input(config["metal"]["pin"])
    binfull = GPIO.input(config["binful"]["proxysensor"]["pin"])

    binfull = not binfull
    metal = not metal
    polybag = not polybag

    if not config["weight"]["active"]:
        tempweight = 6 
    if not config["polybag"]["active"]:
        polybag = False
    if not config["metal"]["active"]:
        metal = False
    if not config["binful"]["active"]:
        binfull = False
    
    data ={
        "weight": tempweight,
        "polybag": polybag,
        "metal": metal,
        "binfull": binfull,
        "bottleStatus": bottleStatus
    }
    return json.dumps(data)

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s

@app.route('/trigger-conv-crusher-data', methods=['POST'])
def trigger_conv_crusher_data():
    global latestThreadLock
    global latestFrame
    data = request.get_json()
    
    if(data["trigger"]):
        print("on triggered")
        current = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        try:
            dataID = data["dataID"]
        except:
            dataID = 0
        itemId = dataID + current + ".jpg"
        bottle = data["bottle"]
        can = data["can"]
        polybag = data["polybag"]
        if(polybag):
            config["data"]["polybags"] += 1
            json.dump(config, open('config.json', 'w'), indent=4)
        if(can):
            config["data"]["cans"] += 1
            json.dump(config, open('config.json', 'w'), indent=4)
            tempFrame = None
            with latestThreadLock:
                tempFrame = latestFrame
            tempFrame = tempFrame.tobytes()
            data = {
                "dataID": dataID,
                "type": "can",
                "time": current,
                "image": tempFrame,
                "imageName": itemId
            }

            csvFile = open("data.csv", "a")
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow([
                data["dataID"],
                data["type"],
                data["time"],
                data["image"]
            ])
            csvFile.close()
            GPIO.output(config["conveyor_fw"]["pin"],True)
            GPIO.output(config["crusher"]["pin"],True)



        if(bottle):
            config["data"]["bottles"] += 1
            json.dump(config, open('config.json', 'w'), indent=4)
            tempFrame = None
            with latestThreadLock:
                tempFrame = latestFrame
            tempFrame = tempFrame.tobytes()
            
            data = {
                "dataID": dataID,
                "type": "bottle",
                "time": current,
                "image": tempFrame,
                "imageName": itemId
            }

            csvFile = open("data.csv", "a")
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow([
                data["dataID"],
                data["type"],
                data["time"],
                data["image"]
            ])
            csvFile.close()
            GPIO.output(config["conveyor_fw"]["pin"],True)
            GPIO.output(config["crusher"]["pin"],True)
        
    else:
        print("off triggered")
        GPIO.output(config["conveyor_fw"]["pin"],False)
        GPIO.output(config["crusher"]["pin"],False)

    return json.dumps({"status": "success"})



def filterData(dataID):
    df = pd.read_csv("data.csv", header=None)
    filteredData = df[df[0] == dataID]
    return filteredData.to_dict('records')

def post_data(data):
    url = "http://13.233.128.164/api/machine/data-images/"
    try:
        requests.post(url, data=data)
        return True
    except Exception as e:
        print(e)
        return False   

@app.route('/upload-images-cloud', methods=['POST'])
def upload_images_cloud():
    data = request.get_json()
    dataID = data["dataID"]
    filteredData = filterData(dataID)
    uploadedData = []
    try:
        s3 = boto3.resource(
            service_name='s3',
            region_name='region_name',
            aws_access_key_id='aws_access_key_id',
            aws_secret_access_key='aws_secret_access_key'
        )
        for data in filteredData:
            s3.Bucket('dataimagestorage').put_object(Key=data[4], Body=data[3])
            dataUrl = "https://dataimagestorage.s3.ap-south-1.amazonaws.com/"+data[4]
            uploadedData.append(
                {
                    "dataID": data[0],
                    "type": data[1],
                    "time": data[2],
                    "image": dataUrl
                }
            )

        if(post_data(uploadedData)):
            print(uploadedData)
            return json.dumps({"status": "success"})
        else:
            
            return json.dumps({"status": "failed"})
    except Exception as e:
        print(e)
        return json.dumps({"status": "failed"})

@app.route('/post-user-data', methods=['POST'])
def post_user_data():
    data = request.get_json()
    try:
        requests.post("http://13.233.128.164/api/machine/machine-data/",data=data)
        return json.dumps({"status": "success"})
    except Exception as e:
        print(e)
        return json.dumps({"status": "failed"})

@app.route('/set-weight-reference', methods=['POST'])
def set_weight_reference():
    data = request.get_json()
    config["weight"]["referenceValue"] = data["reference"]
    json.dump(config, open('config.json', 'w'), indent=4)
    return json.dumps({"status": "success"})

if __name__ == "__main__":
    
    tensor = threading.Thread(target=tensorThread)
    tensor.start()
    weight = threading.Thread(target=weightThread)
    weight.start()
    app.run()