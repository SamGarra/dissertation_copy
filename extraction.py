'''
import zipfile

zip_path = "data.zip"
extract_path = "building_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted successfully")
'''