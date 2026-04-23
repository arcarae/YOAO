"""Human name generator for Sugarscape agents."""

import random
from typing import Optional, Dict, Any

# Diverse pool of real human names from various cultures
HUMAN_NAMES = [
    # Western names
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Logan", "Mia", "Lucas", "Charlotte", "Oliver", "Amelia",
    "James", "Harper", "Benjamin", "Evelyn", "Alexander", "Abigail", "Michael",
    "Emily", "Daniel", "Elizabeth", "Henry", "Sofia", "Jackson", "Avery",
    
    # East Asian names
    "Wei", "Yuki", "Min-jun", "Sakura", "Jian", "Hana", "Chen", "Aiko",
    "Hiro", "Mei", "Kenji", "Ling", "Riku", "Soo-min", "Tao", "Yuna",
    "Koji", "Xiu", "Akira", "Yui", "Jin", "Haru", "Feng", "Nari",
    
    # South Asian names
    "Arjun", "Priya", "Ravi", "Ananya", "Aditya", "Diya", "Rohan", "Neha",
    "Aarav", "Ishaan", "Kavya", "Siddharth", "Aadhya", "Vivaan", "Saanvi",
    "Krishna", "Lakshmi", "Rajesh", "Sita", "Vikram", "Anjali",
    
    # Middle Eastern / Arabic names
    "Omar", "Fatima", "Ali", "Aisha", "Hassan", "Zahra", "Ahmed", "Layla",
    "Yusuf", "Mariam", "Karim", "Noor", "Tariq", "Amina", "Ibrahim", "Sara",
    "Khalid", "Zainab", "Rashid", "Hala",
    
    # African names
    "Kofi", "Amara", "Jabari", "Zuri", "Kwame", "Nia", "Tunde", "Imani",
    "Ayodele", "Sanaa", "Chike", "Kaya", "Ayo", "Amani", "Olufemi", "Zola",
    
    # Latin American names
    "Diego", "Valentina", "Santiago", "Camila", "Mateo", "Isabella", "Alejandro", "Sofia",
    "Gabriel", "Lucia", "Carlos", "Maria", "Miguel", "Carmen", "Rafael", "Rosa",
    "Juan", "Elena", "Luis", "Ana",
    
    # European names (non-English)
    "Lars", "Ingrid", "Marco", "Giulia", "Pierre", "Amelie", "Hans", "Greta",
    "Ivan", "Natasha", "Andre", "Claire", "Stefan", "Petra", "Pavel", "Olga",
    
    # Additional diverse names
    "Kai", "Luna", "River", "Sky", "Phoenix", "Sage", "Atlas", "Nova",
    "Orion", "Iris", "Jasper", "Willow", "Finn", "Hazel", "Rowan", "Ivy",
]


class NameGenerator:
    """Generates unique human names for agents."""
    
    def __init__(self, seed: Optional[int] = None):
        self.used_names = set()
        self.name_counts = {}
        self.rng = random.Random(seed)
        self.available_names = HUMAN_NAMES.copy()
        self.rng.shuffle(self.available_names)
        self.name_index = 0
    
    def generate(self) -> str:
        """Generate a unique human name.
        
        Returns a name from the pool. If all names are used, appends numbers
        to avoid collisions (e.g., "Emma", "Emma 2", "Emma 3").
        """
        # Try to get a fresh name
        if self.name_index < len(self.available_names):
            base_name = self.available_names[self.name_index]
            self.name_index += 1
        else:
            # All names used at least once, pick a random one
            base_name = self.rng.choice(HUMAN_NAMES)
        
        # Check if we need to add a number
        if base_name not in self.name_counts:
            self.name_counts[base_name] = 1
            final_name = base_name
        else:
            # Name already used, increment counter
            self.name_counts[base_name] += 1
            count = self.name_counts[base_name]
            final_name = f"{base_name} {count}"
        
        self.used_names.add(final_name)
        return final_name
    
    def reset(self):
        """Reset the generator (useful for new simulations)."""
        self.used_names.clear()
        self.name_counts.clear()
        self.available_names = HUMAN_NAMES.copy()
        self.rng.shuffle(self.available_names)
        self.name_index = 0

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "used_names": list(self.used_names),
            "name_counts": dict(self.name_counts),
            "available_names": list(self.available_names),
            "name_index": self.name_index,
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.used_names = set(state["used_names"])
        self.name_counts = dict(state["name_counts"])
        self.available_names = list(state["available_names"])
        self.name_index = state["name_index"]
        self.rng.setstate(state["rng_state"])

