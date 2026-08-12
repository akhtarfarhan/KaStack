# src/extractor.py
import pandas as pd
import json
import os
import re
import spacy
from datetime import timedelta

class InformationExtractor:
    def __init__(self):
        print("Loading local NLP extraction model (spaCy)...")
        # Load spaCy for named entity recognition (people, organizations)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please run 'python -m spacy download en_core_web_sm' first.")

    def get_priority(self, text):
        """Determine priority based on keywords."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["important", "urgent", "asap", "deadline"]):
            return "high"
        elif any(word in text_lower for word in ["fyi", "just checking", "just so you know", "quick update"]):
            return "low"
        return "medium"

    def extract_date(self, text, timestamp):
        """Extract explicit dates (YYYY-MM-DD) or resolve relative dates (tomorrow)."""
        # Look for explicit dates in YYYY-MM-DD format
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if date_match:
            return date_match.group(0)
        
        # Look for relative dates
        text_lower = text.lower()
        if "tomorrow" in text_lower:
            return (timestamp + timedelta(days=1)).strftime('%Y-%m-%d')
        if "tonight" in text_lower or "today" in text_lower:
            return timestamp.strftime('%Y-%m-%d')
            
        return None

    def extract_time(self, text):
        """Extract explicit times (e.g., 14:00, 8 PM)."""
        # Match 24-hour time (e.g., 15:00, 09:30)
        time_24_match = re.search(r"\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b", text)
        if time_24_match:
            return time_24_match.group(0)
            
        # Match 12-hour time (e.g., 8 PM, 9 AM)
        time_12_match = re.search(r"\b(1[0-2]|0?[1-9])\s*([AP]M)\b", text, re.IGNORECASE)
        if time_12_match:
            return time_12_match.group(0).upper()
            
        return None

    def extract_person(self, text):
        """Extract a person's name if explicitly mentioned."""
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean up if it caught a prefix like "Maya" from "Call Maya"
                return ent.text
        return None

    def clean_title(self, text):
        """Generate a clean title/description by removing prefixes and trailing info."""
        # Remove common prefixes
        clean_text = re.sub(r"^(Important|FYI|Just checking|Quick update|For today|One more thing|Please note|Can you help\?|Hi,)[—\-:\s]*", "", text, flags=re.IGNORECASE).strip()
        
        # Strip exact dates and times to keep the title clean
        clean_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", clean_text)
        clean_text = re.sub(r"\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b", "", clean_text)
        clean_text = re.sub(r"at\s+in|on\s+at|happens on", "", clean_text, flags=re.IGNORECASE)
        
        # Return the first 50 characters as a clean title description
        return " ".join(clean_text.split()).strip()[:50].strip(".,;")

    def process_extractions(self, raw_csv_path, class_json_path, output_json_path):
        print(f"Loading raw data to resolve timestamps...")
        df = pd.read_csv(raw_csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loading classification report...")
        with open(class_json_path, 'r') as f:
            classifications = json.load(f)
            
        # Create a dictionary for instant lookups
        class_dict = {item['message_id']: item['category'] for item in classifications}
        
        results = []
        task_counter = 1
        event_counter = 1
        
        print("Extracting tasks and events...")
        for _, row in df.iterrows():
            msg_id = row['message_id']
            text = str(row['message'])
            timestamp = row['timestamp']
            
            category = class_dict.get(msg_id, None)
            
            # We only extract info for Tasks (Action Required) and Events (Meeting or Event)
            if category == "action_required":
                item_type = "task"
                item_id = f"TASK_{task_counter:03d}"
                task_counter += 1
            elif category == "meeting_or_event":
                item_type = "event"
                item_id = f"EVNT_{event_counter:03d}"
                event_counter += 1
            else:
                continue # Skip all other categories
                
            # Perform extraction
            results.append({
                "item_id": item_id,
                "type": item_type,
                "title": self.clean_title(text),
                "deadline": self.extract_date(text, timestamp) if item_type == "task" else None,
                "date": self.extract_date(text, timestamp) if item_type == "event" else None,
                "time": self.extract_time(text),
                "person": self.extract_person(text),
                "priority": self.get_priority(text),
                "source_message_id": msg_id
            })

        # Save the results
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Extraction complete! Found {task_counter - 1} tasks and {event_counter - 1} events.")
        print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    # Your exact absolute paths
    RAW_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\data\messages.csv"
    CLASS_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\classification_report.json"
    OUTPUT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\extraction_report.json"
    
    extractor = InformationExtractor()
    extractor.process_extractions(RAW_CSV, CLASS_JSON, OUTPUT_JSON)