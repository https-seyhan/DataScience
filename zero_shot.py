from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

incident_text = "A worker suffered a severe burn due to chemical exposure."

labels = ["Minor Injury", "Major Harm", "Fatality", "Near Miss", "Non-Compliance"]
result = classifier(incident_text, candidate_labels=labels)
print(result)
