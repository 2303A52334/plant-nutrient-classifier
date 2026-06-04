# PlantCare AI: Intelligent Nutrient Diagnostics 🌿

![PlantCare AI Banner](https://raw.githubusercontent.com/2303A52334/plant-nutrient-classifier/main/static/images/banner.png)

## Overview
**PlantCare AI** is a state-of-the-art web application designed to help farmers and gardeners identify nutrient deficiencies in plants using advanced computer vision. Built with **Flask** and **TensorFlow**, it provides instant, AI-driven diagnostics from a simple leaf photo.

### ✨ Key Features
- **Instant Analysis**: Neural network-based classification of leaf health.
- **Modern Dashboard**: Clean, intuitive, and responsive UI for ease of use.
- **High Sensitivity**: Specifically optimized to detect nutrient deficiencies with high recall.
- **Scalable Backend**: Powered by Flask for lightweight, efficient processing.

## 📊 Model Performance
Our custom CNN model has been evaluated against a diverse dataset of plant leaves.

| Metric | Accuracy |
| :--- | :--- |
| **Nutrient Deficiency Detection** | **79%** |
| **Healthy Leaf Validation** | **56%** |
| **Overall Balanced Performance** | **~68%** |

> [!TIP]
> To get the best results, ensure the leaf is well-lit and centered in the photo.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow, Keras, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism design)
- **Image Processing**: Pillow (PIL)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/2303A52334/plant-nutrient-classifier.git
   cd plant-nutrient-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## 📂 Project Structure
```text
├── Healthy/             # Sample healthy leaf images
├── Nutrient/            # Sample nutrient deficient images
├── static/              # CSS, JS, and uploaded images
├── templates/           # HTML templates
├── app.py               # Main Flask application
├── plant_nutrient_classifier.h5  # Pre-trained CNN model
└── requirements.txt     # Python dependencies
```

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
Developed by Kota Sri Priya (https://github.com/2303A52334)
