import os
import sys
import json
import time
import numpy as np
from typing import Optional, Tuple, Dict, List

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ProfileManager:
    """
    Consent-Based Local Memory Store for Face Profiles.
    Stores names, embeddings, and personalized greetings in config/profiles.json.
    Independent of any cloud AI provider.
    """

    def __init__(self, filepath: str = "config/profiles.json"):
        self.filepath = filepath
        self.profiles: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Loads profiles from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
                print(f"📁 ProfileManager: Loaded {len(self.profiles)} registered profile(s).")
            except Exception as e:
                print(f"⚠️ ProfileManager: Error loading profiles: {e}")
                self.profiles = {}
        else:
            self.profiles = {}

    def _save(self):
        """Saves profiles to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ ProfileManager: Error saving profiles: {e}")
            return False

    def register_person(
        self,
        name: str,
        embedding: np.ndarray,
        custom_greeting: Optional[str] = None
    ) -> bool:
        """
        Registers a new person with explicit consent.
        Stores normalized 128D embedding vector.
        """
        if embedding is None:
            return False

        key = name.strip().lower()
        display_name = name.strip()

        # Convert numpy array to list for JSON serialization
        emb_list = [float(v) for v in embedding.flatten()]

        # If already exists, update/append embedding to multi-sample list
        if key in self.profiles:
            existing = self.profiles[key]
            embeddings = existing.get("embeddings", [])
            # Store up to 5 reference samples for varying lighting/angles
            if len(embeddings) >= 5:
                embeddings.pop(0)
            embeddings.append(emb_list)
            existing["embeddings"] = embeddings
            existing["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if custom_greeting:
                existing["custom_greeting"] = custom_greeting
            self.profiles[key] = existing
        else:
            self.profiles[key] = {
                "name": display_name,
                "consent_given": True,
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
                "custom_greeting": custom_greeting or f"Hello {display_name}, wonderful to see you!",
                "embeddings": [emb_list]
            }

        saved = self._save()
        if saved:
            print(f"✅ ProfileManager: Successfully registered '{display_name}'.")
        return saved

    def identify_person(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.363
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Compares query embedding against all saved profiles.
        Returns: (matched_display_name, highest_similarity, custom_greeting)
        """
        if query_embedding is None or not self.profiles:
            return (None, 0.0, None)

        best_name = None
        best_greeting = None
        highest_sim = -1.0

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return (None, 0.0, None)

        q_vec = query_embedding / query_norm

        for key, data in self.profiles.items():
            embeddings = data.get("embeddings", [])
            for ref_list in embeddings:
                ref_vec = np.array(ref_list, dtype=np.float32)
                ref_norm = np.linalg.norm(ref_vec)
                if ref_norm == 0:
                    continue
                r_vec = ref_vec / ref_norm

                # Cosine similarity
                sim = float(np.dot(q_vec, r_vec))
                if sim > highest_sim:
                    highest_sim = sim
                    if sim >= threshold:
                        best_name = data.get("name", key)
                        best_greeting = data.get("custom_greeting")

        if best_name is not None and highest_sim >= threshold:
            # Update last seen timestamp
            key = best_name.lower()
            if key in self.profiles:
                self.profiles[key]["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return (best_name, highest_sim, best_greeting)

        return (None, highest_sim, None)

    def list_people(self) -> List[str]:
        """Returns list of registered display names."""
        return [p.get("name", k) for k, p in self.profiles.items()]

    def delete_person(self, name: str) -> bool:
        """Deletes a person's profile upon request."""
        key = name.strip().lower()
        if key in self.profiles:
            del self.profiles[key]
            self._save()
            print(f"🗑️ ProfileManager: Removed profile for '{name}'.")
            return True
        return False
