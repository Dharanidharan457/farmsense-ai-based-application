farmsense ai implementation

### Step 1: Development Environment Setup
1. Install required software:
   ```bash
   # Install Python dependencies
   pip install tensorflow opencv-python pytorch fastai django djangorestframework psycopg2 django-cors-headers pillow numpy pandas scikit-learn matplotlib paho-mqtt

   # Install React Native environment
   npm install -g expo-cli
   ```

2. Configure cloud resources:
   - Set up AWS/Azure/GCP account for hosting
   - Configure PostgreSQL database
   - Set up Redis for caching

3. Configure version control:
   ```bash
   # Initialize Git repository
   git init
   git add .
   git commit -m "Initial project setup"
   ```

### Step 2: Project Structure Creation
1. Create backend Django project:
   ```bash
   django-admin startproject farmsense_backend
   cd farmsense_backend
   python manage.py startapp api
   python manage.py startapp soil_analysis
   python manage.py startapp pest_detection
   python manage.py startapp weather_forecast
   ```

2. Create frontend React Native app:
   ```bash
   expo init farmsense_mobile
   cd farmsense_mobile
   npm install react-navigation three.js ar.js axios redux react-redux
   ```

#phase2

### Step 1: IoT Sensor Network Setup
1. Hardware components needed:
   - Arduino/ESP32 microcontrollers
   - Soil moisture sensors
   - Soil pH sensors
   - Temperature and humidity sensors
   - LoRaWAN transceivers
   - Solar panels for power

2. Basic sensor node code (Arduino/ESP32):
   ```cpp
   #include <SPI.h>
   #include <LoRa.h>
   #include <Wire.h>
   #include "DHT.h"

   #define DHTPIN 7
   #define DHTTYPE DHT22
   #define MOISTURE_PIN A0
   #define PH_PIN A1
   #define NODE_ID "SENSOR001"

   DHT dht(DHTPIN, DHTTYPE);

   void setup() {
     Serial.begin(9600);
     while (!Serial);
     Serial.println("FarmSense IoT Node");
     
     // Initialize LoRa
     if (!LoRa.begin(915E6)) {
       Serial.println("LoRa initialization failed!");
       while (1);
     }
     
     // Initialize sensors
     dht.begin();
   }

   void loop() {
     // Read sensors
     float humidity = dht.readHumidity();
     float temperature = dht.readTemperature();
     int moistureRaw = analogRead(MOISTURE_PIN);
     int phRaw = analogRead(PH_PIN);
     
     // Convert raw readings to actual values
     float moisturePercent = map(moistureRaw, 0, 1023, 0, 100);
     float phValue = (float)phRaw * 14.0 / 1023.0;
     
     // Create data packet
     String packet = NODE_ID + "," + String(temperature) + "," + 
                     String(humidity) + "," + String(moisturePercent) + "," + 
                     String(phValue);
     
     // Send packet
     LoRa.beginPacket();
     LoRa.print(packet);
     LoRa.endPacket();
     
     Serial.println("Data sent: " + packet);
     delay(300000); // Send data every 5 minutes
   }
   ```

3. LoRaWAN gateway setup:
   - Configure Raspberry Pi with LoRaWAN gateway HAT
   - Install required software:
   ```bash
   # On Raspberry Pi
   sudo apt-get update
   sudo apt-get install python3-pip
   sudo pip3 install paho-mqtt
   ```

4. Gateway data collection script:
   ```python
   import time
   import paho.mqtt.client as mqtt
   from datetime import datetime
   import json
   import requests

   # MQTT Configuration
   MQTT_BROKER = "localhost"
   MQTT_PORT = 1883
   MQTT_TOPIC = "sensor/data"
   
   # API Configuration
   API_ENDPOINT = "https://your-farmsense-api.com/api/sensor-data/"
   API_KEY = "your-api-key"
   
   # MQTT Callbacks
   def on_connect(client, userdata, flags, rc):
       print(f"Connected with result code {rc}")
       client.subscribe(MQTT_TOPIC)
   
   def on_message(client, userdata, msg):
       try:
           payload = msg.payload.decode()
           parts = payload.split(',')
           
           if len(parts) == 5:
               sensor_id, temperature, humidity, moisture, ph = parts
               
               data = {
                   "sensor_id": sensor_id,
                   "temperature": float(temperature),
                   "humidity": float(humidity),
                   "soil_moisture": float(moisture),
                   "soil_ph": float(ph),
                   "timestamp": datetime.now().isoformat()
               }
               
               # Send to API
               headers = {"Authorization": f"Api-Key {API_KEY}"}
               response = requests.post(API_ENDPOINT, json=data, headers=headers)
               
               if response.status_code == 201:
                   print(f"Data from {sensor_id} successfully sent to API")
               else:
                   print(f"Error sending data: {response.status_code}")
           else:
               print(f"Invalid data format: {payload}")
               
       except Exception as e:
           print(f"Error processing message: {e}")
   
   # Setup MQTT client
   client = mqtt.Client()
   client.on_connect = on_connect
   client.on_message = on_message
   
   client.connect(MQTT_BROKER, MQTT_PORT, 60)
   client.loop_forever()
   ```

### Step 2: Drone System Setup
1. Drone hardware requirements:
   - DJI Agricultural drone or similar
   - Multispectral camera attachment
   - Storage for collected imagery

2. Drone flight path planning code:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from shapely.geometry import Polygon, Point
   
   class DroneFlightPlanner:
       def __init__(self, field_coordinates, altitude=30, overlap=0.7):
           """
           Initialize flight planner
           
           Args:
               field_coordinates: List of (lat, lon) tuples defining field boundary
               altitude: Flight altitude in meters
               overlap: Image overlap percentage (0-1)
           """
           self.field = Polygon(field_coordinates)
           self.altitude = altitude
           self.overlap = overlap
           self.camera_fov = 70  # degrees
           
       def calculate_image_footprint(self):
           """Calculate ground coverage of a single image"""
           # Using altitude and FOV to calculate footprint
           view_angle_rad = np.radians(self.camera_fov)
           footprint_width = 2 * self.altitude * np.tan(view_angle_rad / 2)
           return footprint_width
           
       def generate_waypoints(self):
           """Generate drone waypoints for complete field coverage"""
           # Get field bounds
           minx, miny, maxx, maxy = self.field.bounds
           
           # Calculate image footprint
           footprint = self.calculate_image_footprint()
           
           # Calculate step size based on overlap
           step_size = footprint * (1 - self.overlap)
           
           # Generate grid of waypoints
           waypoints = []
           
           # Alternate row direction for efficient path
           row_direction = 1
           y = miny
           
           while y <= maxy:
               if row_direction == 1:
                   x = minx
                   while x <= maxx:
                       point = Point(x, y)
                       if self.field.contains(point):
                           waypoints.append((x, y, self.altitude))
                       x += step_size
               else:
                   x = maxx
                   while x >= minx:
                       point = Point(x, y)
                       if self.field.contains(point):
                           waypoints.append((x, y, self.altitude))
                       x -= step_size
               
               y += step_size
               row_direction *= -1
           
           return waypoints
           
       def visualize_flight_path(self, waypoints):
           """Visualize the flight path"""
           field_x, field_y = self.field.exterior.xy
           
           plt.figure(figsize=(10, 10))
           plt.plot(field_x, field_y, 'k-')
           
           waypoint_x = [w[0] for w in waypoints]
           waypoint_y = [w[1] for w in waypoints]
           
           plt.plot(waypoint_x, waypoint_y, 'b-')
           plt.plot(waypoint_x, waypoint_y, 'ro', markersize=2)
           
           plt.title("Drone Flight Path")
           plt.xlabel("Longitude")
           plt.ylabel("Latitude")
           plt.grid(True)
           plt.savefig("flight_path.png")
           plt.show()
   
   # Example usage
   field_coords = [(0, 0), (0, 100), (100, 100), (100, 0)]
   planner = DroneFlightPlanner(field_coords)
   waypoints = planner.generate_waypoints()
   planner.visualize_flight_path(waypoints)
   ```

3. Image processing pipeline setup:
   ```python
   import cv2
   import numpy as np
   from tensorflow.keras.models import load_model
   
   class MultiSpectralProcessor:
       def __init__(self, model_path):
           self.model = load_model(model_path)
           
       def preprocess_image(self, image_path):
           """Preprocess multispectral image for analysis"""
           # Load image
           img = cv2.imread(image_path)
           
           # Extract different spectral bands
           b, g, r = cv2.split(img)
           
           # Calculate NDVI (if NIR band is available)
           # For simulation, we'll use red and blue channels
           ndvi = (r.astype(float) - b.astype(float)) / (r.astype(float) + b.astype(float) + 1e-8)
           
           # Normalize
           ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
           
           # Resize for model input
           resized = cv2.resize(ndvi_normalized, (224, 224))
           
           return resized
           
       def detect_issues(self, image_path):
           """Detect crop issues from multispectral image"""
           processed_img = self.preprocess_image(image_path)
           
           # Prepare for model input
           model_input = np.expand_dims(processed_img, axis=0) / 255.0
           
           # Get predictions
           predictions = self.model.predict(model_input)
           
           # Process results
           # This will vary based on model output format
           return predictions
           
       def generate_health_map(self, images_folder, output_path):
           """Generate crop health map from multiple images"""
           # Implementation depends on how images are stitched together
           # This is a simplified placeholder
           print("Generating health map from images in", images_folder)
           print("Output will be saved to", output_path)
   ```

## Phase 3

### Step 1: Pest and Disease Detection Model
1. Training data preparation:
   ```python
   import os
   import glob
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
   def prepare_training_data(data_dir, labels_file, output_dir):
       """
       Prepare training data for pest detection model
       
       Args:
           data_dir: Directory containing images
           labels_file: CSV file with image_name,label columns
           output_dir: Directory to save processed data
       """
       # Load labels
       labels_df = pd.read_csv(labels_file)
       
       # Create train/val/test splits
       train_df, temp_df = train_test_split(labels_df, test_size=0.3, random_state=42)
       val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
       
       # Save splits
       os.makedirs(output_dir, exist_ok=True)
       train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
       val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
       test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
       
       # Create data generators for augmentation
       train_datagen = ImageDataGenerator(
           rescale=1./255,
           rotation_range=20,
           width_shift_range=0.2,
           height_shift_range=0.2,
           shear_range=0.2,
           zoom_range=0.2,
           horizontal_flip=True,
           fill_mode='nearest'
       )
       
       val_datagen = ImageDataGenerator(rescale=1./255)
       
       # Create generators
       train_generator = train_datagen.flow_from_dataframe(
           dataframe=train_df,
           directory=data_dir,
           x_col="image_name",
           y_col="label",
           target_size=(224, 224),
           batch_size=32,
           class_mode="categorical"
       )
       
       validation_generator = val_datagen.flow_from_dataframe(
           dataframe=val_df,
           directory=data_dir,
           x_col="image_name",
           y_col="label",
           target_size=(224, 224),
           batch_size=32,
           class_mode="categorical"
       )
       
       return train_generator, validation_generator
   ```

2. Model training code:
   ```python
   import tensorflow as tf
   from tensorflow.keras.applications import MobileNetV2
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
   
   def create_pest_detection_model(num_classes):
       """Create pest detection model using transfer learning"""
       # Base model
       base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
       
       # Add custom layers
       x = base_model.output
       x = GlobalAveragePooling2D()(x)
       x = Dense(1024, activation='relu')(x)
       x = Dropout(0.5)(x)
       predictions = Dense(num_classes, activation='softmax')(x)
       
       # Create model
       model = Model(inputs=base_model.input, outputs=predictions)
       
       # Freeze base model layers
       for layer in base_model.layers:
           layer.trainable = False
           
       # Compile model
       model.compile(
           optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy']
       )
       
       return model
   
   def train_model(model, train_generator, validation_generator, epochs=10):
       """Train the model"""
       # Callbacks
       checkpoint = tf.keras.callbacks.ModelCheckpoint(
           'best_model.h5',
           save_best_only=True,
           monitor='val_accuracy'
       )
       
       early_stop = tf.keras.callbacks.EarlyStopping(
           monitor='val_loss',
           patience=3
       )
       
       # Train
       history = model.fit(
           train_generator,
           steps_per_epoch=train_generator.samples // train_generator.batch_size,
           validation_data=validation_generator,
           validation_steps=validation_generator.samples // validation_generator.batch_size,
           epochs=epochs,
           callbacks=[checkpoint, early_stop]
       )
       
       return history, model
   
   # Example usage
   num_classes = 10  # Number of pest/disease classes
   pest_model = create_pest_detection_model(num_classes)
   history, trained_model = train_model(pest_model, train_generator, validation_generator)
   trained_model.save('pest_detection_model.h5')
   ```

3. Model deployment code:
   ```python
   import flask
   from flask import request, jsonify
   import numpy as np
   import tensorflow as tf
   import cv2
   import io
   from PIL import Image

   app = flask.Flask(__name__)

   # Load model
   model = tf.keras.models.load_model('pest_detection_model.h5')

   # Class names
   class_names = ['healthy', 'aphids', 'blackspot', 'blight', 'powdery_mildew', 
                  'rust', 'scab', 'spider_mites', 'leaf_miners', 'bacterial_spot']

   @app.route('/predict', methods=['POST'])
   def predict():
       if 'image' not in request.files:
           return jsonify({'error': 'No image provided'}), 400
           
       # Get image
       image_file = request.files['image']
       image_bytes = image_file.read()
       
       # Convert to OpenCV format
       nparr = np.frombuffer(image_bytes, np.uint8)
       img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       img = cv2.resize(img, (224, 224))
       img = img / 255.0
       img = np.expand_dims(img, axis=0)
       
       # Make prediction
       predictions = model.predict(img)
       predicted_class = np.argmax(predictions[0])
       confidence = float(predictions[0][predicted_class])
       
       # Return result
       return jsonify({
           'prediction': class_names[predicted_class],
           'confidence': confidence,
           'all_probabilities': {class_name: float(prob) 
                                for class_name, prob in zip(class_names, predictions[0])}
       })

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
   ```

### Step 2: Weather Forecasting Integration
1. Weather API integration code:
   ```python
   import requests
   import json
   from datetime import datetime, timedelta

   class WeatherService:
       def __init__(self, api_key):
           self.api_key = api_key
           self.base_url = "https://api.openweathermap.org/data/2.5/onecall"
           
       def get_weather_forecast(self, lat, lon, days=7):
           """
           Get weather forecast for location
           
           Args:
               lat: Latitude
               lon: Longitude
               days: Number of days for forecast
               
           Returns:
               Processed weather data
           """
           params = {
               "lat": lat,
               "lon": lon,
               "exclude": "current,minutely,hourly,alerts",
               "units": "metric",
               "appid": self.api_key
           }
           
           response = requests.get(self.base_url, params=params)
           
           if response.status_code == 200:
               data = response.json()
               
               # Process data
               processed_data = []
               for day_data in data["daily"][:days]:
                   date = datetime.fromtimestamp(day_data["dt"]).strftime("%Y-%m-%d")
                   
                   processed_day = {
                       "date": date,
                       "temp_min": day_data["temp"]["min"],
                       "temp_max": day_data["temp"]["max"],
                       "humidity": day_data["humidity"],
                       "wind_speed": day_data["wind_speed"],
                       "precipitation": day_data.get("rain", 0),
                       "weather_main": day_data["weather"][0]["main"],
                       "weather_description": day_data["weather"][0]["description"]
                   }
                   
                   processed_data.append(processed_day)
                   
               return processed_data
           else:
               raise Exception(f"Error fetching weather data: {response.status_code}")
               
       def calculate_growing_conditions(self, weather_data, crop_type):
           """
           Calculate growing condition score based on weather and crop type
           
           Args:
               weather_data: Processed weather data
               crop_type: Type of crop
               
           Returns:
               Growing condition scores for each day
           """
           # Crop optimal conditions (simplified)
           crop_conditions = {
               "rice": {"temp_min": 20, "temp_max": 30, "humidity": 70, "rain": 10},
               "wheat": {"temp_min": 15, "temp_max": 25, "humidity": 60, "rain": 5},
               "corn": {"temp_min": 18, "temp_max": 28, "humidity": 65, "rain": 7}
           }
           
           # Default to wheat if crop not found
           optimal = crop_conditions.get(crop_type.lower(), crop_conditions["wheat"])
           
           results = []
           for day in weather_data:
               # Calculate temperature score
               temp_avg = (day["temp_min"] + day["temp_max"]) / 2
               temp_score = 100 - min(100, abs(temp_avg - (optimal["temp_min"] + optimal["temp_max"]) / 2) * 5)
               
               # Calculate humidity score
               humidity_score = 100 - min(100, abs(day["humidity"] - optimal["humidity"]) * 2)
               
               # Calculate rain score
               rain_score = 100 - min(100, abs(day.get("precipitation", 0) - optimal["rain"]) * 10)
               
               # Overall score
               overall_score = (temp_score * 0.5) + (humidity_score * 0.3) + (rain_score * 0.2)
               
               results.append({
                   "date": day["date"],
                   "overall_score": overall_score,
                   "temp_score": temp_score,
                   "humidity_score": humidity_score,
                   "rain_score": rain_score,
                   "recommendation": self._get_recommendation(overall_score)
               })
               
           return results
           
       def _get_recommendation(self, score):
           """Get recommendation based on score"""
           if score >= 80:
               return "Excellent growing conditions"
           elif score >= 60:
               return "Good growing conditions"
           elif score >= 40:
               return "Average conditions, monitor crops"
           else:
               return "Poor conditions, take protective measures"
   ```

## Phase4 twin technology

### Step 1: Backend Data Integration
1. Django models for data storage:
   ```python
   # soil_analysis/models.py
   from django.db import models
   from django.contrib.gis.db import models as gis_models
   
   class Farm(models.Model):
       name = models.CharField(max_length=100)
       owner = models.ForeignKey('auth.User', related_name='farms', on_delete=models.CASCADE)
       location = gis_models.PolygonField()
       area = models.FloatField(help_text="Area in hectares")
       created_at = models.DateTimeField(auto_now_add=True)
       
       def __str__(self):
           return self.name
   
   class Field(models.Model):
       farm = models.ForeignKey(Farm, related_name='fields', on_delete=models.CASCADE)
       name = models.CharField(max_length=100)
       crop_type = models.CharField(max_length=100)
       planting_date = models.DateField()
       expected_harvest_date = models.DateField()
       location = gis_models.PolygonField()
       area = models.FloatField(help_text="Area in hectares")
       
       def __str__(self):
           return f"{self.farm.name} - {self.name}"
   
   class SensorData(models.Model):
       sensor_id = models.CharField(max_length=50)
       field = models.ForeignKey(Field, related_name='sensor_data', on_delete=models.CASCADE)
       timestamp = models.DateTimeField()
       location = gis_models.PointField()
       temperature = models.FloatField()
       humidity = models.FloatField()
       soil_moisture = models.FloatField()
       soil_ph = models.FloatField()
       
       class Meta:
           ordering = ['-timestamp']
           
   # pest_detection/models.py
   from django.db import models
   from soil_analysis.models import Field
   
   class PestDetection(models.Model):
       field = models.ForeignKey('soil_analysis.Field', related_name='pest_detections', on_delete=models.CASCADE)
       image = models.ImageField(upload_to='pest_images/')
       detected_issue = models.CharField(max_length=100)
       confidence = models.FloatField()
       location = models.CharField(max_length=100)
       timestamp = models.DateTimeField(auto_now_add=True)
       recommendations = models.TextField()
       
       def __str__(self):
           return f"{self.field.name} - {self.detected_issue}"
   ```

2. API endpoints:
   ```python
   # api/urls.py
   from django.urls import path, include
   from rest_framework.routers import DefaultRouter
   from . import views
   
   router = DefaultRouter()
   router.register(r'farms', views.FarmViewSet)
   router.register(r'fields', views.FieldViewSet)
   router.register(r'sensor-data', views.SensorDataViewSet)
   router.register(r'pest-detections', views.PestDetectionViewSet)
   router.register(r'weather-forecasts', views.WeatherForecastViewSet)
   
   urlpatterns = [
       path('', include(router.urls)),
       path('farm-health-overview/<int:farm_id>/', views.FarmHealthOverview.as_view()),
       path('pest-detection-upload/', views.PestDetectionUpload.as_view()),
   ]
   ```

### Step 2: Digital Twin Visualization
1. Frontend visualization components:
   ```javascript
   // DigitalTwinView.js
   import React, { useState, useEffect } from 'react';
   import { View, StyleSheet } from 'react-native';
   import MapView, { Polygon, Marker } from 'react-native-maps';
   import { getFieldData, getSensorData } from '../api/farmApi';
   
   const DigitalTwinView = ({ farmId }) => {
     const [fieldData, setFieldData] = useState([]);
     const [sensorData, setSensorData] = useState([]);
     const [loading, setLoading] = useState(true);
     const [selectedField, setSelectedField] = useState(null);
     
     useEffect(() => {
       const loadData = async () => {
         setLoading(true);
         try {
           const fields = await getFieldData(farmId);
           setFieldData(fields);
           
           // Get sensor data for first field or selected field
           const fieldId = selectedField?.id || (fields.length > 0 ? fields[0].id : null);
           if (fieldId) {
             const sensors = await getSensorData(fieldId);
             setSensorData(sensors);
           }
         } catch (error) {
           console.error("Error loading farm data:", error);
         } finally {
           setLoading(false);
         }
       };
       
       loadData();
     }, [farmId, selectedField]);
     
     const getSensorColor = (value) => {
       // Logic to determine color based on sensor value
       if (value > 70) return 'green';
       if (value > 50) return 'yellow';
       return 'red';
     };
     
     return (
       <View style={styles.container}>
         <MapView
           style={styles.map}
           initialRegion={{
             latitude: fieldData[0]?.centerLat || 0,
             longitude: fieldData[0]?.centerLng || 0,
             latitudeDelta: 0.01,
             longitudeDelta: 0.01,
           }}
         >
           {fieldData.map(field => (
             <Polygon
               key={field.id}
               coordinates={field.coordinates}
               strokeColor="#000"
               fillColor={field.health > 70 ? "rgba(0,128,0,0.5)" : 
                         field.health > 50 ? "rgba(255,255,0,0.5)" : 
                         "rgba(255,0,0,0.5)"}
               tappable
               onPress={() => setSelectedField(field)}
             />
           ))}
           
           {sensorData.map(sensor => (
             <Marker
               key={sensor.id}
               coordinate={{ latitude: sensor.latitude, longitude: sensor.longitude }}
               title={`Sensor ${sensor.sensor_id}`}
               description={`Moisture: ${sensor.soil_moisture}%, pH: ${sensor.soil_ph}`}
               pinColor={getSensorColor(sensor.soil_moisture)}
             />
           ))}
         </MapView>
         
         {selectedField && (
           <View style={styles.infoPanel}>
             <Text style={styles.fieldName}>{selectedField.name}</Text>
             <Text>Crop: {selectedField.crop_type}</Text>
             <Text>Health Index: {selectedField.health}/100</Text>
             <Text>Soil Moisture: {selectedField.avgMoisture}%</Text>
             <Text>Recent Issues: {selectedField.recentIssues}</Text>
           </View>
         )}
       </View>
     );
   };
   
   const styles = StyleSheet.create({
     container: {
       flex: 1,
     },
     map: {
       width: '100%',
       height: '70%',
     },
     infoPanel: {
       padding: 15,
       backgroundColor: '#fff',
       borderTopWidth: 1,
       borderColor: '#ddd',
     },
     fieldName: {
       fontSize: 18,
       fontWeight: 'bold',
       marginBottom: 5,
     },
   });
   
   export default DigitalTwinView;
   ```

## Phase 5: Mobile App Development

### Step 1: AR Interface Development
1. React Native AR implementation:
   ```javascript
   // ARViewScreen.js
   import React, { useState, useEffect } from 'react';
   import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
   import { Camera } from 'expo-camera';
   import * as Location from 'expo-location';
   import { fetchNearbyData } from '../api/farmApi';
   
   const ARViewScreen = () => {
     const [hasPermission, setHasPermission] = useState(null);
     const [location, setLocation] = useState(null);
     const [heading, setHeading] = useState(null);
     const [nearbyData, setNearbyData] = useState([]);
     
     useEffect(() => {
       (async () => {
         // Request camera permissions
         const { status: cameraStatus } = await Camera.requestPermissionsAsync();
         
         //