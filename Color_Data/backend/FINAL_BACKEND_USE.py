import main
import warnings
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", category=FutureWarning)

image_path = r"C:\Users\sanka\OneDrive\Pictures\Screenshots\Screenshot (45).png"

analyzer = main.ImageColorAnalyzer(image_path)
color = analyzer.analyze_image()
# print(color)
