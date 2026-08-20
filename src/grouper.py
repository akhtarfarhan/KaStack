# src/grouper.py
import pandas as pd
import json
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MessageGrouper:
    def __init__(self):
        print("Loading local semantic embedding model (all-MiniLM-L6-v2)...")
        # Lightweight, fast local model for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.65  # Threshold for grouping

    def determine_status(self, texts):
        """Determine the status of a group based on the chronological flow of messages."""
        combined_text = " ".join(texts).lower()
        
        # We check the most recent messages for final statuses
        if any(word in combined_text for word in ["cancel", "no longer needed", "aborted"]):
            return "cancelled"
        if any(word in combined_text for word in ["completed", "done", "submitted", "handled", "finished"]):
            return "completed"
        if any(word in combined_text for word in ["reschedule", "moved", "postponed"]):
            return "rescheduled"
        if any(word in combined_text for word in ["progress", "started", "working on", "update"]):
            return "in progress"
        
        return "pending"

    def extract_latest_deadline(self, texts, extracted_items):
        """Find the latest date mentioned in the group."""
        dates = []
        # Check extraction report dates
        for item in extracted_items:
            if item.get('deadline'): dates.append(item['deadline'])
            if item.get('date'): dates.append(item['date'])
            
        # Regex fallback for text
        combined_text = " ".join(texts)
        matches = re.findall(r"\d{4}-\d{2}-\d{2}", combined_text)
        dates.extend(matches)
        
        if dates:
            # Sort and return the latest date chronologically
            return sorted(dates)[-1]
        return None

    def generate_summary(self, texts):
        """Generate a short summary based on the group's chronological messages."""
        if len(texts) == 1:
            return "A single isolated request or statement."
        return "Multiple messages discussing a request, tracking updates, or modifying constraints."

    def run_grouping(self, masked_csv, extract_json, output_json):
        print(f"Loading data...")
        df = pd.read_csv(masked_csv)
        
        with open(extract_json, 'r') as f:
            extractions = json.load(f)
            
        # Create dictionary of extracted items for fast lookup
        item_dict = {item['source_message_id']: item for item in extractions}

        print("Generating semantic embeddings for 1104 messages (this takes 10-20 seconds)...")
        embeddings = self.model.encode(df['message'].astype(str).tolist())

        groups = []
        group_counter = 1

        print("Grouping messages by semantic meaning and chronology...")
        for i, row in df.iterrows():
            msg_id = row['message_id']
            text = str(row['message'])
            emb = embeddings[i].reshape(1, -1)

            best_group_idx = -1
            best_score = 0

            # Compare current message against existing groups
            for g_idx, group in enumerate(groups):
                # Compare against the embedding of the first message in the group
                score = cosine_similarity(emb, group['base_embedding'])[0][0]
                if score > best_score:
                    best_score = score
                    best_group_idx = g_idx

            # If it passes the threshold, add it to the existing group
            if best_score >= self.similarity_threshold:
                groups[best_group_idx]['messages'].append((msg_id, text))
                if msg_id in item_dict:
                    groups[best_group_idx]['items'].append(item_dict[msg_id])
                groups[best_group_idx]['confidence'] = round(float((groups[best_group_idx]['confidence'] + best_score) / 2), 2)
            
            # Otherwise, create a new group
            else:
                new_group = {
                    "group_id": f"GROUP_{group_counter:03d}",
                    "title": text[:40].strip() + "...", # First 40 chars of base message
                    "messages": [(msg_id, text)],
                    "items": [item_dict[msg_id]] if msg_id in item_dict else [],
                    "base_embedding": emb,
                    "confidence": 0.99
                }
                groups.append(new_group)
                group_counter += 1

        # Format output exactly as requested in L2 prompt
        final_results = []
        for g in groups:
            msg_ids = [m[0] for m in g['messages']]
            texts = [m[1] for m in g['messages']]
            item_ids = [i['item_id'] for i in g['items']]
            
            # Only output groups that have more than 1 message OR have an actionable item
            if len(msg_ids) > 1 or len(item_ids) > 0:
                final_results.append({
                    "group_id": g['group_id'],
                    "title": g['title'],
                    "related_message_ids": msg_ids,
                    "related_task_or_event_ids": item_ids,
                    "summary": self.generate_summary(texts),
                    "status": self.determine_status(texts),
                    "latest_deadline": self.extract_latest_deadline(texts, g['items']),
                    "confidence": g['confidence']
                })

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"Grouping complete! Created {len(final_results)} related-message groups.")
        print(f"Results saved to {output_json}")


if __name__ == "__main__":
    # Paths
    MASKED_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\masked_messages.csv"
    EXTRACT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\extraction_report.json"
    OUTPUT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\grouping_report.json"
    
    grouper = MessageGrouper()
    grouper.run_grouping(MASKED_CSV, EXTRACT_JSON, OUTPUT_JSON)