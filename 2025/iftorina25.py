from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.layers import DepthwiseConv2D
import cv2  # Install opencv-python
import numpy as np
import time
import random
import paho.mqtt.client as mqtt
#import ssl



######################## Naudotojo kintamieji ###############################################################################################

broker = "serverio_adresas"        # reikalinga pakeisti. Serverio adresas pateiktas aprasyme, 14psl. 2 punktas
port = porto_numeris               # reikalinga pakeisti, nurodomas skaičiumi

valdiklis="valdiklio_pavadinimas"  # reikalinga pakeisti
irenginys="!šaldytuvas!"           # reikalinga pakeisti

model_dir = "disko_vieta_kur_yra_modelis/"            # reikalinga pakeisti
model_name = "keras_modelio_pavadinimas"              # reikalinga pakeisti


# Atpažinto objekto etikete, kuri turi įjungti šviestuvą
atpazinto_objekto_etikete_on = "objekto_vardas_arba_pavadinimas_1"    # reikalinga pakeisti

# Atpažinto objekto etikete, kuri turi išjungti šviestuvą
atpazinto_objekto_etikete_off = "objekto_vardas_arba_pavadinimas_2"   # reikalinga pakeisti


######################## Naudotojo kintamieji ###############################################################################################



# Publish a message
client_id = f'python-mqtt-{random.randint(0, 1000)}'
model_label = "labels.txt"

#komanda = "ON"
komanda_on = "ON"
komanda_off = "OFF"

# Pagalbiniai kintamieji
prev_atpazintas_objektas = ""
prev_atpazinto_objekto_nr = -1

topic = valdiklis+'/'+irenginys

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
#model = load_model("keras_Model.h5", compile=False)

# Define a custom DepthwiseConv2D class without the groups parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove the 'groups' parameter if it exists
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove the groups parameter
        super().__init__(**kwargs)

# Create a dictionary of custom objects to pass to the load_model function
custom_objects = {
    'DepthwiseConv2D': CustomDepthwiseConv2D,
}

# Load the model with the custom object
try:
    model = load_model(model_dir + model_name, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Load the labels
class_names = open(model_dir + model_label, "r").readlines()

# Define MQTT connection

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected successfully")
    else:
        print(f"Connection failed with code {rc}")

def send_command(topic, komanda, qos=1):
    result = client.publish(topic, komanda, qos)
    status = result[0]  
    return status




client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id, protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.connect(broker, port, keepalive=60)
client.loop_start()



# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
#################################################### Programos pagrindinio ciklo pradžia #########################################################################
while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    obj_atpazinimo_patikimumas = int(str(np.round(confidence_score * 100))[:-2])
    atpazintas_objektas = class_name.split()[1]
    atpazinto_objekto_nr = int(class_name.split()[0])
    
    print("Atpažintas objektas: ", atpazintas_objektas.replace("\n",""), " ir jo nr.:", atpazinto_objekto_nr, " patikimumas:", obj_atpazinimo_patikimumas, "%")
    
    # Atpažinta
    if (atpazintas_objektas.strip() != prev_atpazintas_objektas.strip()) and obj_atpazinimo_patikimumas >= 110:
        prev_atpazintas_objektas = atpazintas_objektas
        
        if atpazintas_objektas.strip() == atpazinto_objekto_etikete_on.strip():
            komanda = komanda_on
        elif atpazintas_objektas.strip() == atpazinto_objekto_etikete_off:
            komanda = komanda_off
        else:
            komanda = atpazintas_objektas
            
        status = send_command(topic, komanda)
        if status == 0:
            print(f"Message sent to topic `{topic}` \n")
        else:
            print(f"Failed to send message to topic `{topic}` \n")
        
    

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
    time.sleep(1)
#################################################### Programos pagrindinio ciklo pabaiga #########################################################################    
    
camera.release()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
