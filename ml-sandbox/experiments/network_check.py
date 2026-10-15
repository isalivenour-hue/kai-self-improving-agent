import urllib.request
with urllib.request.urlopen('https://example.com', timeout=10) as r:
    print('internet_status', r.status)
    print('internet_bytes', len(r.read(128)))
