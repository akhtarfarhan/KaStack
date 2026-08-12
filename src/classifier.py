import pandas as pd
import json
import os
from transformers import pipeline

class LocalMessageClassifier:
    def __init__(self):
        print("Initializing Hugging Face zero-shot model...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        self.categories = [
            "Action Required",
            "Meeting or Event",
            "Personal Information",
            "General Information",
            "Promotional"
        ]

    def generate_reason(self, category, text):
        text_lower = str(text).lower()
        if category == "Action Required":
            return "The sender is prompting the user to complete a task or review an item."
        elif category == "Meeting or Event":
            return "The message references scheduling, temporal markers, or locations for an event."
        elif category == "Personal Information":
            return "The message contains user-specific preferences, identity markers, or internal profiles."
        elif category == "Promotional":
            if "code" in text_lower or "save" in text_lower or "off" in text_lower:
                return "The message contains discount codes or promotional marketing language."
            return "The message is attempting to sell a service or product."
        else:
            return "The message contains standard informational updates or FYI statements."

    def process_messages(self, input_csv, output_json):
        print(f"Loading masked dataset from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        results = []
        total_rows = len(df)
        
        print(f"Starting classification for {total_rows} messages...")
        
        for index, row in df.iterrows():
            msg_id = row['message_id']
            text = str(row['message'])
            
            # FAST-PATH: If the masker already flagged it, it's Sensitive.
            if "******" in text:
                category = "Sensitive Information"
                confidence = 0.99
                reason = "Message contains highly sensitive identifiers masked by the privacy firewall."
            
            # SLOW-PATH: Run the Hugging Face model
            else:
                prediction = self.classifier(text, self.categories)
                # Ensure the category string matches the exact requested JSON format
                category_raw = prediction['labels'][0]
                category = category_raw.lower().replace(" ", "_")
                confidence = round(prediction['scores'][0], 2)
                reason = self.generate_reason(category_raw, text)

            results.append({
                "message_id": msg_id,
                "category": category,
                "confidence": confidence,
                "reason": reason
            })
            
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1}/{total_rows} messages...")

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Classification complete! Results saved to {output_json}")

if __name__ == "__main__":
    # Your exact absolute paths
    INPUT_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\masked_messages.csv"
    OUTPUT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\classification_report.json"
    
    classifier = LocalMessageClassifier()
    classifier.process_messages(INPUT_CSV, OUTPUT_JSON)