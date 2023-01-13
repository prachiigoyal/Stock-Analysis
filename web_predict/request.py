import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'name':'NVAX'})

print(r.json())