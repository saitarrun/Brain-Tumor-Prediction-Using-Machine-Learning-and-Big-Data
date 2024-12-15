import os
import subprocess

# Install required libraries
subprocess.check_call(["pip", "install", "tensorflow", "opencv-python-headless"])

# Import dependencies
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import cv2
import numpy as np
import tensorflow as tf

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ImagePreprocessingAndPrediction") \
    .getOrCreate()

# Load TensorFlow model from GCS
model_path = "gs://braintumor_dataset/models/brain_tumor_model.h5"
model = tf.keras.models.load_model(model_path)

# Define preprocess and predict function
def preprocess_and_predict(image_path):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img_resized = cv2.resize(img, (150, 150)) / 255.0
            img_array = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_array)
            return "Cancer" if prediction[0][0] > 0.5 else "Non Cancer"
        else:
            return "Error: Image not found"
    except Exception as e:
        return f"Error: {str(e)}"

# Register UDF
@pandas_udf(StringType())
def predict_udf(image_paths):
    return image_paths.apply(preprocess_and_predict)

# Load image paths
dataset_path = "gs://braintumor_dataset/"
image_paths_rdd = spark.sparkContext.wholeTextFiles(dataset_path).keys()

# Convert to DataFrame
schema = StructType([StructField("image_path", StringType(), True)])
image_paths_df = spark.createDataFrame(image_paths_rdd.map(lambda x: (x,)), schema=schema)

# Apply predictions
predicted_df = image_paths_df.withColumn("prediction", predict_udf(col("image_path")))

# Save predictions to GCS
output_path = "gs://braintumor_dataset/predictions/"
predicted_df.write.json(output_path)

print("Predictions complete. Results saved to GCS.")
