import urllib.request
import urllib.error

urls = [
    'https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run/health',
    'https://jatinsheoran2412--argus-stream-a-api-argusapi-fastapi-app.modal.run/health'
]

for url in urls:
    try:
        print(f"Testing: {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=60)
        print(f"  Result: {res.getcode()} (SUCCESS)")
        print(f"  Body: {res.read().decode('utf-8')}\n")
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}\n")
    except Exception as e:
        print(f"  Connection Error: {e}\n")
