# 🧠 Brain Tumor Prediction Using Machine Learning and Big Data

A scalable, cloud-based solution for classifying brain MRI scans as cancerous or non-cancerous using advanced machine learning, computer vision, and big data technologies. This project leverages Google Cloud Platform (GCP), Apache Spark, TensorFlow, and Streamlit to deliver a robust and automated end-to-end pipeline.

---

## 📌 Overview

This project tackles early detection of brain tumors through medical image classification. It implements distributed image preprocessing, feature extraction, deep learning model training, and real-time deployment. The solution is designed to be scalable, secure, and easily maintainable.

- **Cloud-native ML pipeline** with GCP services
- **Preprocessing at scale** using Apache Spark on Google Dataproc
- **High-accuracy CNN model** built using TensorFlow
- **Streamlit dashboard** for real-time tumor classification
- **Target accuracy**: ~95% with AUC-ROC evaluation

---

## 🔧 Tech Stack

- **Languages**: Python, Shell/Bash  
- **Frameworks & Libraries**: TensorFlow, Keras, OpenCV, scikit-learn, Matplotlib, Seaborn  
- **Big Data Tools**: Apache Spark (PySpark), Google Dataproc  
- **Cloud Services**: Google Cloud Storage, Cloud Run, Cloud Functions, BigQuery  
- **DevOps**: Docker  
- **Deployment**: Streamlit

---

## 🧠 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Structure**: Organized into `Cancer/` and `Non-Cancer/` folders  
- **Image Formats**: TIFF, JPEG, PNG  
- **Dataset Size**: ~3,000 images

---

## 🧪 Pipeline Overview

### 1. Data Ingestion (Google Cloud Storage)
- Create GCS buckets with `Cancer/` and `Non-Cancer/` folders
- Upload images using `google-cloud-storage` library
- Convert unsupported formats (TIFF) to JPEG/PNG
- Automate ingestion with **Cloud Functions**
- Enable **GCS versioning** and validate upload integrity

### 2. Preprocessing (Apache Spark)
- Resize images to 128x128 resolution
- Normalize pixel values to [0, 1]
- Use **Spark on Dataproc** for distributed preprocessing
- Save processed outputs to GCS for training

### 3. Feature Engineering
- Flatten images for traditional ML input
- Normalize and reduce dimensions using PCA
- Extract deep features using pre-trained CNN (e.g., ResNet)
- Store features in BigQuery or GCS

### 4. Model Training
- Input: 128x128x3 MRI images  
- Architecture: Custom CNN  
- Loss Function: Binary Cross Entropy  
- Optimizer: Adam  
- Metrics: Precision, Recall, F1-score, AUC  
- Environment: TensorFlow with GPU/TPU support  
- Split: Train/Validation/Test from preprocessed GCS data

### 5. Deployment
- Build an interactive UI with **Streamlit**
- Integrate the trained CNN for real-time inference
- Deploy on **Google Cloud Run** for scalability and accessibility

### 6. Testing & Validation
- Use **Grad-CAM** to highlight regions influencing predictions
- Perform end-to-end validation: ingestion → prediction
- Conduct stress tests and latency evaluations

### 7. Results
- Accuracy: ~95%  
- Precision: 93%  
- Recall: 94%  
- AUC-ROC: 0.96  
- Visual insights via Matplotlib, Seaborn, and Grad-CAM overlays

### 8. Maintenance & Scalability
- Enable model retraining with new data
- Monitor dashboard uptime and model accuracy
- Plan for HIPAA-compliant extensions and HIS/EMR integration

---

## 🛡️ Compliance & Security

- Secure data uploads with IAM and role-based access
- Version control enabled for all ingestion workflows
- Designed to align with best practices for healthcare data environments

---

## 🎓 Academic Context

This project was developed as part of the **Advanced Database & Big Data Systems** coursework. It demonstrates real-world deployment of machine learning and big data concepts for healthtech applications.

---

## 🤝 Contributions

Feel free to fork, open issues, or submit pull requests. Collaboration is welcome!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

