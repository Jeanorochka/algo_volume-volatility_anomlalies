import json

CONFIG_PATH = "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

tickers = cfg.get("tickers", {})

for t, params in tickers.items():
    # --- TP / SL ---
    params["take_profit_pct"] = 0.0025   # +0.25%
    params["stop_loss_pct"]   = -0.0015  # -0.15%

    # --- Чуть "раньше" вход: смягчаем критерий аномальности ---
    old_vol = float(params.get("vol_z_entry", 2.5))
    old_ret = float(params.get("ret_z_entry", 1.4))

    # уменьшаем пороги на 10% (чуть-чуть, не радикально)
    new_vol = old_vol * 0.9
    new_ret = old_ret * 0.9

    params["vol_z_entry"] = round(new_vol, 3)
    params["ret_z_entry"] = round(new_ret, 3)

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(
    f"Готово: для {len(tickers)} тикеров обновлены "
    "TP=0.25%, SL=-0.15%, vol_z/ret_z снижены на 10%."
)
