import requests
from bs4 import BeautifulSoup

URL = "https://debene.dev"

def check_adsense_head(url):
    res = requests.get(url, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    head = soup.head

    if not head:
        print("❌ <head> tag not found.")
        return

    ads_script = head.find("script", src=lambda s: s and "adsbygoogle.js" in s)
    meta_tag = head.find("meta", attrs={"name": "google-adsense-account"})

    if ads_script:
        print("✅ AdSense script tag is present in <head>.")
    else:
        print("❌ AdSense script tag NOT found in <head>.")

    if meta_tag:
        print(f"✅ Meta tag present: {meta_tag}")
    else:
        print("❌ Meta tag for google-adsense-account NOT found in <head>.")

check_adsense_head(URL)