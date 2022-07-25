import requests

record_id = "6894041"  # Zenodo ID (https://zenodo.org/record/6894041)

r = requests.get(f"https://zenodo.org/api/records/{record_id}")
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [f['key'] for f in r.json()['files']]

print(r.status_code)
print(download_urls)

for filename, url in zip(filenames, download_urls):
    print("Downloading:", filename)
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)