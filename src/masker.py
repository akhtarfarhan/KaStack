import pandas as pd
import re
import json
import os

class PrivacyMasker:
    def __init__(self):
        # Strict regex patterns based on the dataset
        self.patterns = {
            "bank_or_payment_details": [
                (r"\b(?:\d{4}\s){3}\d{4}(?:-\d+)?\b", "high", "do_not_store"), 
                (r"\b\d{12}(?:-\d+)?\b", "high", "do_not_store")
            ],
            "one_time_password": [
                (r"\b\d{6}(?:-\d+)?\b", "high", "do_not_store")
            ],
            "authentication_token": [
                (r"\btok_demo_[A-Z0-9]+(?:-\d+)?\b", "high", "do_not_store"),
                (r"\bRC-\d{2}-[A-Z]{2}-\d{2}(?:-\d+)?\b", "high", "do_not_store")
            ],
            "password": [
                (r"(?i)password\s+([A-Za-z0-9#-]+)", "high", "do_not_store")
            ],
            "personal_identification": [
                (r"\bID-\d{4}-[A-Z]{2}(?:-\d+)?\b", "medium", "do_not_send_to_external_service")
            ],
            "private_address_contact": [
                (r"\b\d{5}\s\d{5}(?:-\d+)?\b", "medium", "ask_for_confirmation"),
                (r"(?i)address is\s+([^,.]+(?:,\s*[^,.]+)*)", "medium", "do_not_send_to_external_service")
            ]
        }

    def process_message(self, message_id, text):
        masked_text = str(text)
        report = None
        
        for sensitivity_type, pattern_rules in self.patterns.items():
            for pattern, risk, action in pattern_rules:
                matches = re.findall(pattern, masked_text)
                if matches:
                    for match in matches:
                        target = match if isinstance(match, str) else match[0]
                        masked_text = masked_text.replace(target, "******")
                    
                    if not report:
                        report = {
                            "message_id": message_id,
                            "sensitivity_type": sensitivity_type,
                            "risk": risk,
                            "masked_text": masked_text, 
                            "recommended_action": action
                        }
        
        if report:
            report["masked_text"] = masked_text
            
        return masked_text, report

    def run_pipeline(self, input_csv, masked_csv, report_json):
        print(f"Loading raw data from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        reports = []
        masked_messages = []
        
        print("Scanning and masking messages...")
        for _, row in df.iterrows():
            masked_text, report = self.process_message(row['message_id'], row['message'])
            masked_messages.append(masked_text)
            
            if report:
                reports.append(report)
                
        df['message'] = masked_messages
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(masked_csv), exist_ok=True)
        
        # Save safe data and report
        df.to_csv(masked_csv, index=False)
        with open(report_json, 'w') as f:
            json.dump(reports, f, indent=2)
            
        print(f"Masked dataset saved to: {masked_csv}")

if __name__ == "__main__":
    # Your exact absolute paths
    INPUT_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\data\messages.csv"
    MASKED_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\masked_messages.csv"
    REPORT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\sensitive_report.json"
    
    masker = PrivacyMasker()
    masker.run_pipeline(INPUT_CSV, MASKED_CSV, REPORT_JSON)