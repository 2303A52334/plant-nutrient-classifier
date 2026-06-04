import requests
import os

url = 'http://127.0.0.1:5000/'
healthy_files = ['1.jpg', '10.jpg', '100.jpg', '101.jpg', '102.jpg']
nutrient_files = ['1.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg']

def test_predictions(folder, files, expected):
    results = []
    print(f"\nTesting {expected} images:")
    for filename in files:
        path = os.path.join(folder, filename)
        with open(path, 'rb') as f:
            r = requests.post(url, files={'file': f})
            # The app returns "Status: <prediction>"
            if expected in r.text:
                results.append(True)
                print(f"  {filename}: CORRECT")
            else:
                # Find what it actually predicted
                pred = "Unknown"
                if "Healthy" in r.text and expected == "Nutrient Deficiency":
                    pred = "Healthy"
                elif "Nutrient Deficiency" in r.text and expected == "Healthy":
                    pred = "Nutrient Deficiency"
                results.append(False)
                print(f"  {filename}: WRONG (Predicted: {pred})")
    return results

healthy_results = test_predictions('Healthy', healthy_files, 'Healthy')
nutrient_results = test_predictions('Nutrient', nutrient_files, 'Nutrient Deficiency')

print("\n--- Summary ---")
print(f"Healthy Accuracy: {sum(healthy_results)}/{len(healthy_results)}")
print(f"Nutrient Accuracy: {sum(nutrient_results)}/{len(nutrient_results)}")
