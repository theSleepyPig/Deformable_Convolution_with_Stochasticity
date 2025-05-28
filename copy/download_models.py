# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")