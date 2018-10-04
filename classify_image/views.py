import io
import os
import cv2   
import numpy as np

from base64 import b64decode
import tensorflow as tf
from PIL import Image
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

MAX_K = 10

TF_GRAPH = "{base_path}/inception_model/graph.pb".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
TF_LABELS = "{base_path}/inception_model/labels.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


impath=""

def load_graph():
    sess = tf.Session()
    with tf.gfile.FastGFile(TF_GRAPH, 'rb') as tf_graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tf_graph.read())
        tf.import_graph_def(graph_def, name='')
    label_lines = [line.rstrip() for line in tf.gfile.GFile(TF_LABELS)]
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    return sess, softmax_tensor, label_lines


SESS, GRAPH_TENSOR, LABELS = load_graph()


@csrf_exempt
def classify_api(request):
    data = {"success": False}

    if request.method == "POST":
        tmp_f = NamedTemporaryFile()

        if request.FILES.get("image", None) is not None:
            image_request = request.FILES["image"]
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.save(tmp_f, image.format)
            impath=tmp_f
        elif request.POST.get("image64", None) is not None:
            base64_data = request.POST.get("image64", None).split(',', 1)[1]
            plain_data = b64decode(base64_data)
            tmp_f.write(plain_data)
            impath=tmp_f

        classify_result = tf_classify(tmp_f, int(request.POST.get('k', MAX_K)))
        tmp_f.close()

        if classify_result:
            data["success"] = True
            data["confidence"] = {}
            for res in classify_result:
                data["confidence"][res[0]] = float(res[1])

    return JsonResponse(data)


def classify(request):
    blue_color='yes'
    context={
        'blue_color':checkBlueColor(impath)
        }
    return render(request, 'classify.html', context)


# noinspection PyUnresolvedReferences
def tf_classify(image_file, k=MAX_K):
    result = list()

    image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()

    predictions = SESS.run(GRAPH_TENSOR, {'DecodeJpeg/contents:0': image_data})
    predictions = predictions[0][:len(LABELS)]
    top_k = predictions.argsort()[-k:][::-1]
    for node_id in top_k:
        label_string = LABELS[node_id]
        score = predictions[node_id]
        result.append([label_string, score])

    return result


def checkBlueColor(imgPath):
    frame = cv2.imread(imgPath)
    
    img = frame

    #convert BGR to HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #define Range of colors
    blue_lower=np.array([99,115,150],np.uint8)
    blue_upper=np.array([110,255,255],np.uint8)


    blue=cv2.inRange(hsv,blue_lower,blue_upper)
    kernal = np.ones((5 ,5), "uint8")
    #print("kernal done")

    #find blue contours
    blue=cv2.dilate(blue,kernal)
    res1=cv2.bitwise_and(img, img, mask = blue)
    (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    count=0
    ImgBlueStat="no"
    #print file path for no blue detect. 
    if(contours==[]):
        print(imgPath)
        return "None"
    else:
        for pic, contour in enumerate(contours):
            
            area = cv2.contourArea(contour)
            
            if(area>300):
                
                if(count==0):

                    #print fpath and yes only once.
                    #print(imgPath, "yes")
                    count=count+1
                    return "Yes"
                
                x,y,w,h = cv2.boundingRect(contour)	
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,0,0))
