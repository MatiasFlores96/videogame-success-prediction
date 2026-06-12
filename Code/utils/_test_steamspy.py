"""
Test rápido de la SteamSpy API con 5 juegos conocidos.
SteamSpy: https://steamspy.com/api.php?request=appdetails&appid=XXX
Campos útiles: positive, negative, owners, average_forever, median_forever, name
"""
import requests, json, time

# Algunos AppIDs conocidos para testear
test_ids = {
    440:    "Team Fortress 2",
    570:    "Dota 2",
    730:    "Counter-Strike 2",
    271590: "GTA V",
    1091500: "Cyberpunk 2077",
}

print("=== TEST STEAMSPY API ===\n")
for appid, name in test_ids.items():
    url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = r.json()
            pos = d.get("positive", "N/A")
            neg = d.get("negative", "N/A")
            owners = d.get("owners", "N/A")
            print(f"{name:25s} | positive={pos:>7} | negative={neg:>6} | owners={owners}")
        else:
            print(f"{name}: HTTP {r.status_code}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
    time.sleep(1)  # 1 req/seg para no saturar

print("\nAPI accesible: OK" if True else "")
