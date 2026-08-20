# src/priority_engine.py
import pandas as pd
import json
import os

class PriorityEngine:
    def __init__(self):
        print("Initializing Priority & Action Engine...")

    def determine_priority(self, item, group, texts):
        signals = []
        priority = "medium"
        confidence = 0.75
        reason = "Standard actionable item with no critical modifiers."

        # Extract group context
        status = group.get("status", "pending")
        latest_deadline = group.get("latest_deadline")
        
        # Combine all texts in this group's thread to search for context clues
        combined_text = " ".join(texts).lower()

        # 1. Status Check (Completed / Cancelled drop priority to Low)
        if status == "completed":
            return "low", "Task has been confirmed as completed by a follow-up message.", ["status_completed"], 0.95
        if status == "cancelled":
            return "low", "Event or task was cancelled in a subsequent message.", ["status_cancelled"], 0.95

        # 2. Urgency and Follow-up Keywords
        has_urgent_keyword = any(w in combined_text for w in ["urgent", "asap", "critical", "immediately"])
        if has_urgent_keyword:
            signals.append("urgent_keyword")
            priority = "high"
            confidence = 0.85
            reason = "Message thread contains urgency keywords demanding immediate attention."

        # 3. Deadline Proximity & Changing Deadlines
        if latest_deadline:
            signals.append("has_deadline")
            # Heuristic: Check if follow-ups indicate impending deadlines
            if "today" in combined_text or "tomorrow" in combined_text:
                signals.append("deadline_imminent")
                
                if has_urgent_keyword or len(texts) > 1:
                    priority = "critical"
                    confidence = 0.94
                    reason = "The submission deadline is imminent and a follow-up message marks it as urgent."
                    signals.append("urgent_follow_up")
                else:
                    priority = "high"
                    confidence = 0.88
                    reason = "Deadline is approaching quickly."
        
        # 4. Sensitivity Routing
        if "******" in combined_text:
            signals.append("contains_sensitive_data")
            if priority in ["low", "medium"]:
                priority = "high"
                reason = "Contains sensitive data requiring elevated attention and secure handling."

        # Fallback signal
        if not signals:
            signals.append("standard_priority")

        return priority, reason, signals, confidence

    def run(self, masked_csv, extract_json, group_json, output_json):
        # Load the safe text messages
        df = pd.read_csv(masked_csv)
        msg_dict = dict(zip(df['message_id'], df['message']))
        
        # Load the reports
        with open(extract_json, 'r') as f: extractions = json.load(f)
        with open(group_json, 'r') as f: groups = json.load(f)

        # Map each item_id to its respective group for O(1) lookup
        item_to_group = {}
        for g in groups:
            for item_id in g.get('related_task_or_event_ids', []):
                item_to_group[item_id] = g

        results = []
        print(f"Evaluating dynamic priority for {len(extractions)} tasks and events...")
        
        for item in extractions:
            item_id = item['item_id']
            source_msg_id = item['source_message_id']
            
            group = item_to_group.get(item_id, {})
            # Get all message IDs in this thread, or just the original if ungrouped
            related_msg_ids = group.get('related_message_ids', [source_msg_id])
            
            # Fetch the actual masked text for all messages in the thread
            texts = [str(msg_dict.get(m_id, "")) for m_id in related_msg_ids]

            # Calculate Priority
            priority, reason, signals, conf = self.determine_priority(item, group, texts)

            # Format to exactly match the L2 Example JSON requirement
            results.append({
                "message_id": source_msg_id,
                "item_id": item_id,
                "priority": priority,
                "reason": reason,
                "signals": signals,
                "confidence": conf
            })

        # Save results
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Priority Engine complete! Priority assigned to {len(results)} items.")
        print(f"Saved to {output_json}")

if __name__ == "__main__":
    # Absolute paths
    MASKED_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\masked_messages.csv"
    EXTRACT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\extraction_report.json"
    GROUP_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\grouping_report.json"
    OUTPUT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\priority_report.json"
    
    engine = PriorityEngine()
    engine.run(MASKED_CSV, EXTRACT_JSON, GROUP_JSON, OUTPUT_JSON)