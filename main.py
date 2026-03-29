import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import os
import glob
import csv
from datetime import datetime


# --- 1. GOVERNANCE (Validace a Logování) ---
class Governance:
    def __init__(self, log_file="result/governance_log.csv"):
        self.log_file = log_file
        if not os.path.exists("result"):
            os.makedirs("result")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "action_type", "message"])

    def validate_input(self, image_path):
        if not os.path.exists(image_path):
            self.log_action("INPUT_REJECTED", f"Path not found: {image_path}")
            return False, "File does not exist"
        try:
            img = Image.open(image_path)
            img.verify()

            if img.format not in ['JPEG', 'PNG', 'BMP', 'GIF', 'WEBP']:
                return False, f"Unsupported format: {img.format}"
            return True, None
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    def check_decision_confidence(self, result, threshold):
        # Kontrola, zda máme klíč 'image_path'
        img_name = os.path.basename(result.get('image_path', 'unknown'))

        if result["confidence"] < threshold or result["needs_review"]:
            self.log_action("DECISION_FLAGGED", f"Low conf for {img_name}")
            return False, "Flagged for review"
        self.log_action("DECISION_APPROVED", f"Approved {img_name}")
        return True, None

    def ethical_content_check(self, result):
        return True, None

    def log_action(self, action_type, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, action_type, message])


# --- 2. HLAVNÍ GENERÁTOR ---
class EshopContentGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.governance = Governance()

        print(f"Startuji na zařízení: {self.device}")

        print("Načítám BLIP model (Offline mód)...")
        # Zkusíme to načíst z cache, abychom nepotřebovali internet
        try:
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                   local_files_only=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                              local_files_only=True).to(self.device)
        except Exception:
            print(" V cache nic není")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base").to(self.device)

        # B) Konfigurace kategorií
        self.class_labels = [
            "dress", "handbag", "hat", "high heels", "jacket",
            "jeans", "running shoes", "sneakers", "sunglasses",
            "t-shirt", "watch"
        ]

        # Mapping: Detailní kategorie -> Hlavní kategorie
        self.category_hierarchy = {
            "dress": "Clothing", "jeans": "Clothing", "jacket": "Clothing", "t-shirt": "Clothing",
            "sneakers": "Shoes", "high heels": "Shoes", "running shoes": "Shoes",
            "handbag": "Accessories", "sunglasses": "Accessories", "hat": "Accessories", "watch": "Accessories"
        }

        #Načtení vlastního ResNet18 modelu
        print("Načítám ResNet18...")
        self.custom_classifier = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        num_ftrs = self.custom_classifier.fc.in_features
        self.custom_classifier.fc = nn.Linear(num_ftrs, len(self.class_labels))

        self.custom_classifier = self.custom_classifier.to(self.device)

        # Načtení natrénovaných vah
        if os.path.exists("student_model.pth"):
            # map_location řeší situaci, kdy jsi trénoval na GPU ale teď jsi na CPU (nebo naopak)
            state_dict = torch.load("student_model.pth", map_location=self.device)
            self.custom_classifier.load_state_dict(state_dict)
            print(f"Vlastní model načten! (Kategorií: {len(self.class_labels)})")
        else:
            print("Nenalezen soubor 'student_model.pth'")

        # Transformace pro vstup do sítě (musí být stejné jako při tréninku)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_with_custom_model(self, image, threshold=0.4):
        """Používá ResNet pro určení detailní kategorie"""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.custom_classifier.eval()  # Přepnutí do eval módu

        with torch.no_grad():
            outputs = self.custom_classifier(image_tensor)
            probs = torch.softmax(outputs, dim=1)

        confidence, predicted_idx = torch.max(probs, 1)
        confidence = confidence.item()

        if confidence < threshold:
            return "Uncategorized", confidence

        return self.class_labels[predicted_idx], confidence

    def generate_description(self, image):
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.caption_model.generate(**inputs, max_new_tokens=50)
        return self.caption_processor.decode(out[0], skip_special_tokens=True)

    def marketing_polish(self, raw_caption, sub_category):
        if sub_category == "Uncategorized":
            return "Check out our latest arrival. High quality and style guaranteed."

        # Vylepšený text: Zmíníme konkrétní typ produktu (např. 'sneakers')
        return f"Discover our new {sub_category}. {raw_caption.capitalize()}. A perfect choice for your style!"

    def process_message(self, message):
        """MCP Handler"""
        image_path = message.get("image_path")
        threshold = message.get("threshold", 0.4)

        # 1. Validace
        valid, msg = self.governance.validate_input(image_path)
        if not valid: return {"status": "error", "message": msg}

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return {"status": "error", "message": "Image load failed"}

        # 2. Klasifikace (ResNet)
        # sub_category bude např. "sneakers"
        sub_category, conf = self.classify_with_custom_model(image, threshold)

        # 3. Hierarchie: Z "sneakers" uděláme "Shoes - sneakers"
        if sub_category in self.category_hierarchy:
            main_category = self.category_hierarchy[sub_category]
            full_category_name = f"{main_category} - {sub_category}"
        else:
            main_category = "Uncategorized"
            full_category_name = sub_category

        # 4. Popis (BLIP)
        technical_desc = self.generate_description(image)

        # 5. Marketing (posíláme tam detailní kategorii pro lepší text)
        marketing_desc = self.marketing_polish(technical_desc, sub_category)

        result = {
            "image_path": image_path,
            "image": os.path.basename(image_path),
            "category": full_category_name,
            "sub_category": sub_category,
            "confidence": round(conf, 4),
            "needs_review": sub_category == "Uncategorized",
            "technical_desc": technical_desc,
            "marketing_desc": marketing_desc,
            "model_used": "ResNet18_Detailed_v2"
        }

        # Logování výsledku
        self.governance.check_decision_confidence(result, threshold)

        return {"status": "success", "data": result}


# --- USAGE ---
if __name__ == "__main__":
    bot = EshopContentGenerator()

    # Najde všechny obrázky ve složce 'test_images'
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
        print("Vytvořena složka 'test_images'.")

    image_files = []
    for ext in ["jpg", "jpeg", "png", "webp"]:
        image_files.extend(glob.glob(f"test_images/*.{ext}"))

    if not image_files:
        print("Ve složce 'test_images' nejsou žádné fotky.")
    else:
        print(f"Nalezeno {len(image_files)} obrázků ke zpracování...")
        results_list = []

        for img_path in image_files:
            print(f"--- Zpracovávám: {img_path} ---")
            msg = {"type": "process_image", "image_path": img_path}
            response = bot.process_message(msg)

            if response["status"] == "success":
                print(json.dumps(response["data"], indent=2))
                results_list.append(response["data"])
            else:
                print(f"CHYBA: {response['message']}")

        # Uložení výsledků do JSON
        if results_list:
            with open("result/final_output.json", "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)
            print("Výsledky uloženy v 'result/final_output.json'")