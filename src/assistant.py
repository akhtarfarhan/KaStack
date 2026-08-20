# src/assistant.py
import pandas as pd
import json
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class IntelligentAssistant:
    def __init__(self):
        print("Loading local Semantic RAG model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_databases(self, group_json, priority_json, sensitive_json, extract_json):
        # Load all our structured L1 & L2 knowledge bases
        with open(group_json, 'r') as f: self.groups = json.load(f)
        with open(priority_json, 'r') as f: self.priorities = json.load(f)
        with open(sensitive_json, 'r') as f: self.sensitive = json.load(f)
        with open(extract_json, 'r') as f: self.extractions = json.load(f)
        
        # Pre-compute group embeddings for semantic search
        self.group_texts = [g['title'] + " " + g.get('summary', '') for g in self.groups]
        self.group_embeddings = self.model.encode(self.group_texts) if self.group_texts else []

    def retrieve_answer(self, query):
        q_lower = query.lower()
        
        # 1. Privacy Routing Queries (Blocked / Confirmation)
        if "blocked" in q_lower or "external processing" in q_lower:
            blocked = [s for s in self.sensitive if s['recommended_action'] in ['do_not_send_to_external_service', 'do_not_store']]
            if blocked:
                msg_ids = list(set([b['message_id'] for b in blocked if "DEMO_" in b['message_id']]))
                return {
                    "query": query,
                    "answer": f"Found {len(msg_ids)} demo messages that must be blocked from external processing due to highly sensitive data.",
                    "supporting_message_ids": msg_ids[:5], # Return top 5 for brevity
                    "group_id": None,
                    "reason": "Messages contain sensitive entities marked with 'do_not_store' or 'do_not_send' by the Privacy Firewall."
                }

        if "confirmation" in q_lower:
            confirm = [s for s in self.sensitive if s['recommended_action'] == 'ask_for_confirmation']
            if confirm:
                return {
                    "query": query,
                    "answer": "These messages require human confirmation before processing due to medium-risk sensitive data (e.g., addresses or phone numbers).",
                    "supporting_message_ids": [c['message_id'] for c in confirm][:5],
                    "group_id": None,
                    "reason": "Privacy engine flagged these items with 'ask_for_confirmation'."
                }

        # 2. Priority & Task Queries (Critical / Rescheduled / Completed)
        if "critical" in q_lower:
            critical = [p for p in self.priorities if p['priority'] == 'critical']
            if critical:
                return {
                    "query": query,
                    "answer": f"There are {len(critical)} tasks currently marked as critical priority.",
                    "supporting_message_ids": [c['message_id'] for c in critical][:5],
                    "group_id": None,
                    "reason": "Filtered the Priority Engine output for 'critical' items."
                }
                
        if "completed" in q_lower or "cancelled" in q_lower:
            matched_groups = [g for g in self.groups if g['status'] in ['completed', 'cancelled']]
            if matched_groups:
                return {
                    "query": query,
                    "answer": f"Found {len(matched_groups)} tasks/meetings that have been completed or cancelled.",
                    "supporting_message_ids": matched_groups[0]['related_message_ids'],
                    "group_id": matched_groups[0]['group_id'],
                    "reason": f"Group status evaluates to {matched_groups[0]['status']} based on chronological follow-up messages."
                }

        if "rescheduled" in q_lower:
            rescheduled = [g for g in self.groups if g['status'] == 'rescheduled']
            if rescheduled:
                g = rescheduled[0]
                return {
                    "query": query,
                    "answer": f"A meeting was rescheduled. The latest identified deadline/time is {g.get('latest_deadline', 'Unknown')}.",
                    "supporting_message_ids": g['related_message_ids'],
                    "group_id": g['group_id'],
                    "reason": "Group status updated to 'rescheduled' based on follow-up updates."
                }

        # 3. Specific ID Lookups (e.g., DEMO_016)
        match = re.search(r'(MSG_\d+|DEMO_\d+)', query)
        if match:
            target_id = match.group(1)
            for g in self.groups:
                if target_id in g['related_message_ids']:
                    return {
                        "query": query,
                        "answer": f"The latest status of the item referenced by {target_id} is: {g['status'].upper()}.",
                        "supporting_message_ids": g['related_message_ids'],
                        "group_id": g['group_id'],
                        "reason": f"Message {target_id} belongs to this group, which tracks the chronological status."
                    }

        # 4. Semantic Search (Catch-all for content questions like "compliance form")
        query_emb = self.model.encode([query])
        scores = cosine_similarity(query_emb, self.group_embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        # Strict hallucination prevention: If semantic match is weak, say evidence is unavailable
        if best_score < 0.35:
            return {
                "query": query,
                "answer": "Sufficient evidence is unavailable in the processed datasets to answer this query.",
                "supporting_message_ids": [],
                "group_id": None,
                "reason": f"Semantic relevance score ({best_score:.2f}) was below the confidence threshold."
            }
        
        # Good semantic match found
        best_group = self.groups[best_idx]
        return {
            "query": query,
            "answer": f"Based on semantic search, this relates to group: '{best_group['title']}'. The current status is {best_group['status']}.",
            "supporting_message_ids": best_group['related_message_ids'],
            "group_id": best_group['group_id'],
            "reason": f"Semantic search matched this group with a relevance score of {best_score:.2f}."
        }

    def process_queries(self, queries_csv, output_json):
        print(f"Loading queries from {queries_csv}...")
        df_queries = pd.read_csv(queries_csv)
        
        results = []
        for _, row in df_queries.iterrows():
            q = row['query']
            print(f"Processing query: {q}")
            res = self.retrieve_answer(q)
            results.append(res)
            
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nAssistant processed {len(results)} queries.")
        print(f"Results saved to {output_json}")

if __name__ == "__main__":
    # Absolute paths
    QUERIES_CSV = r"C:\Users\farha\Desktop\KaStack\project_root\data\l2_demo_queries.csv"
    GROUP_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\grouping_report.json"
    PRIORITY_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\priority_report.json"
    SENSITIVE_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\sensitive_report.json"
    EXTRACT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\extraction_report.json"
    OUTPUT_JSON = r"C:\Users\farha\Desktop\KaStack\project_root\outputs\assistant_responses.json"
    
    assistant = IntelligentAssistant()
    assistant.load_databases(GROUP_JSON, PRIORITY_JSON, SENSITIVE_JSON, EXTRACT_JSON)
    assistant.process_queries(QUERIES_CSV, OUTPUT_JSON)