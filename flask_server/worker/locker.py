import urllib.request
import urllib.parse

from flask_server.common.config import ESP32_IDR

def send_locker(locker_id):
    base_url = ESP32_IDR
    number = int(locker_id)
    params = urllib.parse.urlencode({"locker": number})

    url = f"{base_url}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            status = response.status
            result = response.read().decode()
            print("Response:", result)
            if status == 200:
                return True
            else:
                return False

    except Exception as e:
        print("Error:", e)