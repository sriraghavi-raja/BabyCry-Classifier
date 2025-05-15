# 👶 Baby Cry Detector - Backend

A Python-based backend system that detects baby cries and classifies them into different categories (like **hunger**, **pain**, **tiredness**, etc.) using a trained machine learning model. Built with Flask or FastAPI, this backend powers the inference logic for the Baby Cry Classifier app.

---

## 🚀 Features

- 🔊 Detects baby cries from audio input
- 🧠 Classifies cries as hunger, pain, etc.
- 🎯 REST API for easy integration with frontend/mobile apps
- 📁 Predict using `.wav` audio files
- ✅ Lightweight, modular, and scalable

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Framework**: Flask / FastAPI
- **ML/DL**: TensorFlow / Keras
- **Audio Processing**: Librosa, Pydub
- **Deployment**: Localhost / Cloud ready

---

## 📁 Project Structure

baby-detector/
├── backend/
│ ├── app/
│ │ ├── main.py # API routes
│ │ ├── model/ # Trained ML model
│ │ ├── utils.py # Helper functions
│ ├── audio_samples/ # Sample cry audios
│ ├── requirements.txt # Python dependencies
│ ├── README.md # Project documentation
│ └── .gitignore # Files to ignore in git



---

## 📸 Screenshots

📍 Add screenshots in a folder named `screenshots/` and update the paths below:

> ### 📌 API Running in Terminal  
> ![API Running](screenshots/api-terminal.png)

> ### 📌 Postman Test Example  
> ![Postman](screenshots/postman-test.png)

> ### 📌 Prediction Output  
> ![Prediction](screenshots/prediction-result.png)

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sriraghavi-raja/Baby-Cry-Classifier.git
cd Baby-Cry-Classifier/backend


### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# OR
source venv/bin/activate     # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend API

If you're using **FastAPI**:

```bash
uvicorn app.main:app --reload
```

Or if you're using **Flask**:

```bash
python app/main.py
```

---

## 🔍 API Endpoints

| Method | Endpoint   | Description                         |
| ------ | ---------- | ----------------------------------- |
| POST   | `/predict` | Upload baby cry audio & get results |
| GET    | `/health`  | Check if backend is running         |

---

## 🎯 Example: Prediction Request

Using **cURL**:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@audio_samples/test.wav"
```

Using **Postman**:

* Set method: POST
* URL: `http://localhost:8000/predict`
* Body → form-data → Key = `file`, Value = Upload `.wav` file

---

## ✅ Sample Output

```json
{
  "status": "success",
  "prediction": "Pain Cry",
  "confidence": 0.91
}
```

---

## 📌 TODO

* [ ] Improve model accuracy
* [ ] Add Docker support
* [ ] Connect frontend interface
* [ ] Add real-time audio streaming

---

## 🙋‍♀️ Author

**Sriraghavi Raja**
🔗 [GitHub](https://github.com/sriraghavi-raja)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```

---

Let me know if you're using **Flask** or **FastAPI**, and I’ll tailor the run instructions and endpoint formats accordingly. I can also help you automatically generate the `requirements.txt` file if needed.
```
