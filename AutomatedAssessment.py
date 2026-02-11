# For OCR
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import torch
# For Langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# First Load the needed model and move it to the GPU
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)
# Load YOLO model
yolo_model = YOLO("model.pt")

# Image Process and OCR
def segment_and_ocr(image_path): 
    # Load image with PIL
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    # Run YOLO inference
    results = yolo_model(image_path)

    full_text = ""

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

        # Sort boxes by y1 to maintain line order
        boxes = sorted(boxes, key=lambda b: b[1])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop line image
            line_image = pil_image.crop((x1, y1, x2, y2))

            # TrOCR OCR
            pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = trocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            full_text += text.strip() + " "

    return full_text.strip()

### NEEED Edit
student_answer = segment_and_ocr()
model_answer  = input("Enter the Model Answer:  ")
### NEEED Edit 

# LLM and RAG part
os.environ["GOOGLE_API_KEY"] = "AIzaSyBBljFa18cGMmPgBvbDtnl0doFJ_X17ePQ"

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

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(model_answer=model_answer, student_answer=student_answer)

try:
    score = float(result.strip().replace('%', ''))
    print(f"Similarity Score: {score:.2f}%")

    if score > 90:
        print("✅ Full mark 🏆")
    elif score >= 75:
        print("🎉 Three quarters ✨")
    elif score > 50:
        print("😊 Half mark 👍")
    elif score > 25:
        print("🤏 Quarter mark")
    else:
        print("❌ Wrong answer 💔")

except Exception as e:
    print("⚠ Could not parse score:", repr(result))
    print("Error:", e)
