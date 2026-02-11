from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)

# Load YOLO model
yolo_model = YOLO("model.pt")  # Update path as needed

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBBljFa18cGMmPgBvbDtnl0doFJ_X17ePQ"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Prompt Template
template = """
Compare the Student Answer to the Model Answer—rate their similarity from 0 to 100.
Give the score only (as a number), based purely on whether they mean the same thing.
Ignore grammar, structure, or word differences.

Model Answer:
{model_answer}

Student Answer:
{student_answer}
"""

prompt = PromptTemplate(
    input_variables=["model_answer", "student_answer"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def segment_and_ocr(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    results = yolo_model(image_path)
    full_text = ""

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        boxes = sorted(boxes, key=lambda b: b[1])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            line_image = pil_image.crop((x1, y1, x2, y2))
            pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = trocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            full_text += text.strip() + " "

    return full_text.strip()

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    try:
        extracted_text = segment_and_ocr(file_path)
        os.remove(file_path)
        return jsonify({"extracted_text": extracted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    student_answer = data.get("student_answer")
    model_answer = data.get("model_answer")

    if not student_answer or not model_answer:
        return jsonify({"error": "Missing student or model answer"}), 400

    try:
        result = chain.run(model_answer=model_answer, student_answer=student_answer)
        score = float(result.strip().replace('%', ''))
        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)