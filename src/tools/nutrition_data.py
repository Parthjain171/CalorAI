"""Seed nutrition table: the fast path for foods Indian users actually text about.

Values are per one natural serving (one roti, one cup of chai, one plate of
biryani) and are approximate - good enough for calorie tracking, which is a
trend-following exercise, not a lab measurement.

This table exists for latency, not completeness. A hit here costs ~0ms and zero
tokens; a miss falls through to the SQLite cache and then to an LLM estimate.
Roughly 80% of the test traffic lands here.
"""

from __future__ import annotations

from typing import Dict, Tuple

# food -> (serving description, calories, protein_g, carbs_g, fat_g)
SEED_NUTRITION: Dict[str, Tuple[str, float, float, float, float]] = {
    # --- Indian breads & breakfast ---
    "roti": ("1 medium roti", 110, 3.0, 20.0, 2.5),
    "paratha": ("1 plain paratha", 260, 5.0, 32.0, 12.0),
    "aloo paratha": ("1 aloo paratha", 300, 6.0, 40.0, 13.0),
    "naan": ("1 naan", 260, 8.0, 45.0, 5.0),
    "butter naan": ("1 butter naan", 320, 8.0, 46.0, 11.0),
    "puri": ("1 puri", 100, 2.0, 12.0, 5.0),
    "idli": ("1 idli", 58, 2.0, 12.0, 0.3),
    "dosa": ("1 plain dosa", 170, 4.0, 28.0, 5.0),
    "masala dosa": ("1 masala dosa", 290, 6.0, 42.0, 11.0),
    "vada": ("1 medu vada", 140, 4.0, 16.0, 7.0),
    "poha": ("1 plate poha", 250, 5.0, 40.0, 8.0),
    "upma": ("1 plate upma", 230, 6.0, 35.0, 8.0),
    "dhokla": ("100 g dhokla", 160, 6.0, 24.0, 4.0),
    "paav": ("1 pav", 130, 4.0, 25.0, 1.5),
    # --- rice & mains ---
    "rice": ("1 cup cooked rice", 205, 4.2, 45.0, 0.4),
    "jeera rice": ("1 cup jeera rice", 240, 4.5, 44.0, 5.0),
    "curd rice": ("1 bowl curd rice", 250, 7.0, 40.0, 7.0),
    "khichdi": ("1 bowl khichdi", 280, 10.0, 45.0, 6.0),
    "biryani": ("1 plate (~350 g) biryani", 500, 18.0, 65.0, 18.0),
    "chicken biryani": ("1 plate chicken biryani", 550, 26.0, 62.0, 22.0),
    "veg biryani": ("1 plate veg biryani", 450, 10.0, 68.0, 15.0),
    "pulao": ("1 plate pulao", 350, 7.0, 58.0, 10.0),
    # --- dals & curries ---
    "dal": ("1 bowl dal", 150, 9.0, 20.0, 4.0),
    "dal makhani": ("1 bowl dal makhani", 330, 12.0, 28.0, 18.0),
    "rajma": ("1 bowl rajma", 210, 12.0, 30.0, 4.0),
    "rajma chawal": ("1 plate rajma with rice", 415, 16.2, 75.0, 4.4),
    "chole": ("1 bowl chole", 240, 11.0, 32.0, 8.0),
    "sambar": ("1 bowl sambar", 110, 5.0, 15.0, 3.0),
    "paneer curry": ("1 bowl paneer curry", 320, 14.0, 12.0, 25.0),
    "palak paneer": ("1 bowl palak paneer", 280, 13.0, 12.0, 21.0),
    "paneer tikka": ("100 g paneer tikka", 270, 18.0, 6.0, 19.0),
    "butter chicken": ("1 bowl butter chicken", 340, 25.0, 10.0, 22.0),
    "tandoori chicken": ("100 g tandoori chicken", 195, 27.0, 2.0, 8.0),
    "chicken curry": ("1 bowl chicken curry", 260, 24.0, 8.0, 14.0),
    "fish curry": ("1 bowl fish curry", 220, 22.0, 6.0, 12.0),
    "mixed vegetable": ("1 bowl mixed veg sabzi", 150, 4.0, 16.0, 8.0),
    # --- snacks & street food ---
    "samosa": ("1 samosa", 260, 4.0, 30.0, 14.0),
    "pakora": ("100 g pakora", 320, 7.0, 30.0, 19.0),
    "vada pav": ("1 vada pav", 290, 7.0, 42.0, 10.0),
    "pav bhaji": ("1 plate pav bhaji", 400, 9.0, 52.0, 17.0),
    "maggi": ("1 pack maggi", 350, 8.0, 50.0, 13.0),
    "biscuit": ("1 biscuit", 50, 0.7, 7.0, 2.2),
    "namkeen": ("30 g namkeen", 150, 3.5, 16.0, 8.0),
    "chips": ("30 g potato chips", 160, 2.0, 15.0, 10.0),
    # --- eggs, dairy, drinks ---
    "chai": ("1 cup chai with milk & sugar", 90, 2.5, 12.0, 3.0),
    "coffee": ("1 cup milk coffee", 70, 3.0, 8.0, 3.0),
    "black coffee": ("1 cup black coffee", 5, 0.3, 0.0, 0.0),
    "milk": ("1 cup whole milk", 150, 8.0, 12.0, 8.0),
    "curd": ("1 cup curd", 100, 6.0, 8.0, 4.0),
    "raita": ("1 bowl raita", 90, 4.0, 8.0, 4.0),
    "lassi": ("1 glass sweet lassi", 180, 6.0, 26.0, 5.0),
    "buttermilk": ("1 glass chaas", 60, 3.0, 6.0, 2.0),
    "egg": ("1 boiled egg", 78, 6.3, 0.6, 5.3),
    "omelette": ("2-egg omelette", 220, 13.0, 2.0, 17.0),
    "protein shake": ("1 scoop whey in water", 120, 24.0, 3.0, 1.5),
    # --- western & misc ---
    "bread": ("1 slice bread", 80, 2.6, 14.0, 1.0),
    "toast": ("1 buttered toast", 130, 2.8, 14.0, 7.0),
    "oats": ("40 g dry oats", 150, 5.0, 27.0, 3.0),
    "sandwich": ("1 veg sandwich", 250, 8.0, 35.0, 9.0),
    "pizza": ("1 slice pizza", 285, 12.0, 36.0, 10.0),
    "burger": ("1 veg burger", 350, 15.0, 35.0, 17.0),
    "french fries": ("1 medium fries", 340, 4.0, 44.0, 17.0),
    "salad": ("1 bowl green salad", 60, 2.0, 10.0, 1.5),
    "soup": ("1 bowl soup", 110, 4.0, 15.0, 3.0),
    "banana": ("1 banana", 105, 1.3, 27.0, 0.4),
    "apple": ("1 apple", 95, 0.5, 25.0, 0.3),
    "almonds": ("10 almonds", 70, 2.5, 2.5, 6.0),
    "peanuts": ("30 g peanuts", 170, 7.5, 5.0, 14.0),
    "gulab jamun": ("1 gulab jamun", 150, 2.0, 22.0, 6.0),
    "ice cream": ("1 scoop ice cream", 140, 2.5, 17.0, 7.0),
    "chocolate": ("30 g milk chocolate", 160, 2.2, 17.0, 9.0),
}

# Spellings and synonyms that should resolve to a seed entry.
ALIASES: Dict[str, str] = {
    "chapati": "roti",
    "chapathi": "roti",
    "phulka": "roti",
    "rotli": "roti",
    "tea": "chai",
    "chaai": "chai",
    "masala chai": "chai",
    "parantha": "paratha",
    "parotta": "paratha",
    "dahi": "curd",
    "yogurt": "curd",
    "yoghurt": "curd",
    "chaas": "buttermilk",
    "dal fry": "dal",
    "daal": "dal",
    "toor dal": "dal",
    "moong dal": "dal",
    "chana masala": "chole",
    "chhole": "chole",
    "paneer": "paneer curry",
    "shahi paneer": "paneer curry",
    "matar paneer": "paneer curry",
    "boiled egg": "egg",
    "eggs": "egg",
    "anda": "egg",
    "steamed rice": "rice",
    "white rice": "rice",
    "chawal": "rice",
    "fries": "french fries",
    "veg sandwich": "sandwich",
    "whey": "protein shake",
    "curd rice bowl": "curd rice",
    "sabzi": "mixed vegetable",
    "subzi": "mixed vegetable",
}

# Coarse per-serving estimates used only when the LLM estimator is unavailable
# (no API key, network failure). Deliberately blunt - better than refusing to log.
FALLBACK_BY_KEYWORD: Dict[str, Tuple[float, float, float, float]] = {
    "chicken": (250, 24.0, 6.0, 14.0),
    "mutton": (290, 24.0, 4.0, 20.0),
    "fish": (210, 22.0, 5.0, 11.0),
    "paneer": (300, 15.0, 10.0, 23.0),
    "curry": (220, 8.0, 18.0, 12.0),
    "fried": (330, 7.0, 32.0, 19.0),
    "salad": (70, 2.5, 10.0, 2.0),
    "juice": (110, 1.0, 26.0, 0.2),
    "cake": (330, 4.0, 45.0, 15.0),
    "sweet": (200, 3.0, 30.0, 8.0),
    "rice": (210, 4.5, 45.0, 1.0),
    "bread": (90, 3.0, 16.0, 1.5),
    "dal": (160, 9.0, 21.0, 4.0),
}

DEFAULT_FALLBACK: Tuple[float, float, float, float] = (220.0, 7.0, 28.0, 9.0)
