import gradio as gr
import google.generativeai as genai
import gspread
import pandas as pd
import os
import csv
import re
from linear_regression_predictor import predict_cost
from google.oauth2.service_account import Credentials

# ==== KONFIGURASI GOOGLE API & GEMINI ====
SERVICE_ACCOUNT_FILE = 'soy-bazaar-472411-e5-a3884a5d9ba9.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
genai.configure(api_key="AIzaSyA5t8nGEfS0ssZeGkeZTsSGw53XQooaB5k")

model = genai.GenerativeModel("gemini-1.5-flash")

# ==== LOAD DATASET ====
try:
    df = pd.read_csv("dataset_filled.csv", parse_dates=["timestamp"])
except Exception:
    df = pd.read_csv("dataset.csv", parse_dates=["timestamp"])

if os.path.exists("dataset_pln_clean.csv"):
    df_pln = pd.read_csv("dataset_pln_clean.csv")
else:
    df_pln = None

# ==== LOAD DATASET_SMARTHOME ====
if os.path.exists("dataset_smart_home_done.csv"):
    df_smarthome = pd.read_csv(
    "dataset_smart_home_done.csv",
    parse_dates=["time"],
    low_memory=False,
    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S", errors='coerce')
)
else:
    df_smarthome = None

# ==== CHATLOG ====
chatlog_file = "chatlog.csv"
if not os.path.exists(chatlog_file):
    with open(chatlog_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "bot"])

def save_chat(user_msg, bot_msg, filename="chatlog.csv"):
    try:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([user_msg, bot_msg])
    except Exception as e:
        print(f"Error saving chat: {e}")

# ==== FUNGSI DATASET ====
def total_energy_per_device():
    return df.groupby("appliance")["energy_kWh_interval"].sum().sort_values(ascending=False)

def total_cost_per_room():
    return df.groupby("room")["cost_Rp_interval"].sum().sort_values(ascending=False)

# ==== FUNGSI DATASET_SMARTHOME ====
def total_energy_smarthome():
    if df_smarthome is not None:
        device_cols = [col for col in df_smarthome.columns if "[kW]" in col]
        hasil = {}
        for col in device_cols:
            total_kwh = df_smarthome[col].sum()
            hasil[col.replace(" [kW]", "")] = total_kwh
        hasil_sorted = dict(sorted(hasil.items(), key=lambda item: item[1], reverse=True))
        return hasil_sorted
    return None

def total_cost_smarthome():
    if df_smarthome is not None:
        device_cols = [col for col in df_smarthome.columns if "[kW]" in col and col.lower() != "solar [kw]"]
        tarif_per_kwh = 1500  # Sesuaikan dengan tarif PLN
        hasil = {}
        for col in device_cols:
            total_kwh = df_smarthome[col].sum()
            hasil[col.replace(" [kW]", "")] = total_kwh * tarif_per_kwh
        hasil_sorted = dict(sorted(hasil.items(), key=lambda item: item[1], reverse=True))
        return hasil_sorted
    return None

# ==== MAP HARI ====
day_map = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "sabtu": 5, "minggu": 6
}

def extract_prediksi_params(history, curr_message):
    energy, hour, day_str = None, None, None
    pattern = r'(\d+(?:\.\d+)?)\s*kwh'
    hour_pattern = r'jam\s*(\d+)'
    day_pattern = r'hari\s*(\w+)'

    messages = [msg["content"].lower() for msg in history if msg["role"] == "user"]
    messages.append(curr_message.lower())
    full_text = " ".join(messages)

    match = re.search(pattern, full_text)
    if match:
        energy = float(match.group(1))
    match = re.search(hour_pattern, full_text)
    if match:
        hour = int(match.group(1))
    match = re.search(day_pattern, full_text)
    if match:
        day_str = match.group(1)
    return energy, hour, day_str

# ==== CHATBOT LOGIC ====
def chatbot_fn(message, history):
    ans = "Maaf, terjadi error tak terduga."
    msg_lower = message.lower().strip()

    greetings = [
        "hai", "halo", "hello", "hei", "hallo", "assalamualaikum",
        "selamat pagi", "selamat siang", "selamat sore", "selamat malam"
    ]
    thanks = [
        "makasih", "terima kasih", "thanks", "thank you", "trims", "makasi"
    ]
    if msg_lower in greetings:
        ans = "Hai juga! Ada yang bisa saya bantu terkait Smart Energy Management Box atau kelistrikan?"
        save_chat(message, ans)
        return ans
    if msg_lower in thanks:
        ans = "Sama-sama! Jika ada pertanyaan lain seputar Smart Energy Management Box, silakan."
        save_chat(message, ans)
        return ans

    allowed_keywords = [
        "energi", "listrik", "smart energy", "dataset",
        "appliance", "perangkat", "daya", "kwh", "ruangan", "room",
        "konsumsi", "biaya", "hemat", "penghematan", "prediksi", "ramalan",
        "management box", "monitoring", "penggunaan", "ac", "kipas", "lampu", "smarthome"
    ]
    if not any(word in msg_lower for word in allowed_keywords):
        ans = (
            "Maaf, saya hanya dapat menjawab pertanyaan seputar Smart Energy Management Box, "
            "kelistrikan, energi, perangkat listrik, penghematan, monitoring, atau data terkait."
        )
        save_chat(message, ans)
        return ans

    # ==== Jawaban dari DATASET_SMARTHOME ====
    if "smarthome" in msg_lower:
        if "total energi" in msg_lower:
            s = total_energy_smarthome()
            if s is not None:
                ans = "Total konsumsi energi per perangkat di SmartHome (kWh):\n" + "\n".join([f"{a}: {v:.2f}" for a,v in s.items()])
            else:
                ans = "Dataset smarthome tidak tersedia."
            save_chat(message, ans)
            return ans
        if "total biaya" in msg_lower:
            s = total_cost_smarthome()
            if s is not None:
                ans = "Total biaya listrik per perangkat di SmartHome (Rp):\n" + "\n".join([f"{a}: Rp {v:,.2f}" for a,v in s.items()])
            else:
                ans = "Dataset smarthome tidak tersedia."
            save_chat(message, ans)
            return ans
        # Perbaikan logika konsumsi energi perangkat smarthome
        match_sm = re.search(r'konsumsi (energi|listrik) (\w+)', msg_lower)
        if match_sm and df_smarthome is not None:
            appliance = match_sm.group(2)
            try:
                col_candidates = [col for col in df_smarthome.columns if appliance.lower() in col.lower() and "[kW]" in col]
                if col_candidates:
                    s = sum(df_smarthome[col].sum() for col in col_candidates)
                    ans = f"Konsumsi energi {appliance} di SmartHome: {s:.2f} kWh"
                else:
                    ans = f"Tidak ada data konsumsi {appliance} di SmartHome"
            except Exception as e:
                ans = f"Error: {e}"
            save_chat(message, ans)
            return ans

    # ==== Jawaban dari DATASET utama ====
    if "total energi" in msg_lower:
        try:
            s = total_energy_per_device()
            ans = "Total konsumsi energi per perangkat (kWh):\n" + "\n".join([f"{a}: {v:.2f}" for a, v in s.items()])
        except Exception as e:
            ans = f"Error ambil data energi: {e}"
        save_chat(message, ans)
        return ans

    if "total biaya" in msg_lower:
        try:
            s = total_cost_per_room()
            ans = "Total biaya listrik per ruangan (Rp):\n" + "\n".join([f"{a}: Rp {v:,.2f}" for a, v in s.items()])
        except Exception as e:
            ans = f"Error ambil data biaya: {e}"
        save_chat(message, ans)
        return ans

    # Konsumsi energi di dataset utama
    match = re.search(r'konsumsi (energi|listrik) (\w+)', msg_lower)
    if match:
        appliance = match.group(2)
        try:
            s = df[df["appliance"].str.lower() == appliance.lower()]["energy_kWh_interval"].sum()
            if s > 0:
                ans = f"Konsumsi energi {appliance}: {s:.2f} kWh"
            else:
                ans = f"Tidak ada data konsumsi {appliance}"
        except Exception as e:
            ans = f"Error: {e}"
        save_chat(message, ans)
        return ans

    if "prediksi" in msg_lower or "ramalan" in msg_lower:
        energy, hour, day_str = extract_prediksi_params(history, message)
        if energy and hour and day_str:
            weekday = day_map.get(day_str, 0)
            try:
                pred = predict_cost(energy, hour, weekday)
                ans = f"Prediksi biaya {energy} kWh jam {hour} hari {day_str.capitalize()}: Rp {pred:,.2f}"
            except Exception as e:
                ans = f"Error prediksi: {e}"
        else:
            ans = "Lengkapi format: contoh 'prediksi harga listrik 2 kWh jam 19 hari senin'"
        save_chat(message, ans)
        return ans

    # ==== LLM fallback (topik allowed) ====
    try:
        prompt = (
            "Kamu adalah Reno, asisten virtual Smart Energy Management Box.\n"
            "Jawab pertanyaan user hanya jika masih sesuai topik kelistrikan, energi, perangkat listrik, penghematan, monitoring, "
            "atau data Smart Energy Management Box. Jika tidak, tolak secara sopan.\n"
            f"Pertanyaan user: {message}\n"
            "Jawab dengan singkat, jelas, dan berbasis data jika memungkinkan."
        )
        response = model.generate_content(prompt)
        ans = response.text.strip()
    except Exception as e:
        ans = f"Error LLM: {e}"

    save_chat(message, ans)
    return ans

# ==== GRADIO UI ====
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ RenoBot - Smart Energy Management Box Assistant")

    chatbot_ui = gr.Chatbot(
        value=[{"role": "assistant", "content": "Hai, aku Reno. Tanyakan tentang Smart Energy Management Box, kelistrikan, atau data energinya."}],
        label="Chat dengan Reno",
        type="messages"
    )
    msg = gr.Textbox(label="Pesan", placeholder="Tulis pertanyaan tentang Smart Energy Management Box, energi, atau perangkat listrik...")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = chatbot_fn(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot_ui], [msg, chatbot_ui])
    clear.click(lambda: [], None, chatbot_ui, queue=False)

    gr.Examples(
        examples=[
            "Total energi tiap perangkat apa?",
            "Total energi smarthome tiap perangkat?",
            "Total biaya smarthome tiap ruangan?",
            "Konsumsi energi AC berapa?",
            "Konsumsi energi AC di smarthome berapa?",
            "Prediksi harga listrik 2 kWh jam 19 hari senin",
            "Bagaimana cara menghemat listrik di rumah?",
            "Fitur monitoring pada Smart Energy Management Box apa saja?"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch()