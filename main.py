"""BioShorts: автоматическое создание и загрузка короткого видео о полезном продукте.

Шаги: очередь тем -> данные USDA -> сценарий от LLM -> озвучка -> фото с Wikimedia Commons
-> сборка видео (FFmpeg) -> миниатюра -> загрузка на YouTube -> отметка темы как выполненной.
"""

import asyncio
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import edge_tts
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
W, H = 1080, 1920
VOICE = "ru-RU-SvetlanaNeural"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
UA = {"User-Agent": "BioShortsBot/1.0 (personal educational channel)"}
PRIVACY = os.environ.get("PRIVACY", "private")
USDA_KEY = os.environ.get("USDA_KEY", "DEMO_KEY")
LLM_MODEL = "openrouter/free"

# Номера нутриентов по справочнику USDA
NUTRIENTS = {
    "1008": "Калорийность, ккал",
    "1003": "Белки, г",
    "1004": "Жиры, г",
    "1005": "Углеводы, г",
    "1079": "Клетчатка, г",
    "1087": "Кальций, мг",
    "1089": "Железо, мг",
    "1090": "Магний, мг",
    "1092": "Калий, мг",
    "1162": "Витамин C, мг",
}

SYSTEM_PROMPT = """Ты пишешь сценарии коротких видео (YouTube Shorts) на русском языке о полезных растениях и продуктах.
Правила:
- Опирайся на данные USDA, которые тебе переданы. Не выдумывай цифры.
- Не давай медицинских обещаний: не пиши «лечит», «излечивает», «помогает от». Пиши «содержит», «используется в кухне».
- Без музыки, без рекламы, без упоминания людей.
- Сценарий 90–110 слов, 5–7 коротких предложений.
Верни ТОЛЬКО JSON без пояснений с полями:
"title" (до 60 символов, без хештегов),
"script" (текст сценария),
"description" (2–3 предложения),
"tags" (список из 8 слов),
"image_queries" (5 поисковых запросов на английском для фото растения)."""


def load_topic():
    path = ROOT / "topics.json"
    topics = json.loads(path.read_text(encoding="utf-8"))
    for topic in topics:
        if topic.get("status") == "todo":
            return topics, topic
    raise SystemExit("Очередь тем пуста: добавьте новые темы в topics.json")


def save_topics(topics):
    (ROOT / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def usda(query):
    r = requests.get(
        "https://api.nal.usda.gov/fdc/v1/foods/search",
        params={
            "api_key": USDA_KEY,
            "query": query,
            "pageSize": 5,
            "dataType": "Foundation,SR Legacy",
        },
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    foods = r.json().get("foods", [])
    if not foods:
        raise SystemExit(f"USDA: нет данных по запросу '{query}'")
    food = foods[0]
    values = {}
    for n in food.get("foodNutrients", []):
        num = str(n.get("nutrientNumber", ""))
        if num in NUTRIENTS and n.get("value") is not None:
            values[NUTRIENTS[num]] = n["value"]
    return food["description"], food["fdcId"], values


def llm(user_msg):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['LLM_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "BioShorts",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.6,
        },
        timeout=180,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0))
    for key in ("title", "script", "description", "tags", "image_queries"):
        if key not in data:
            raise ValueError(f"в ответе нет поля {key}")
    return data


def ask_llm(user_msg):
    last_error = None
    for attempt in range(3):
        try:
            return llm(user_msg)
        except Exception as error:  # сеть или неверный JSON: повторяем
            last_error = error
            print(f"Попытка LLM {attempt + 1} не удалась: {error}")
    raise SystemExit(f"LLM не вернул корректный ответ: {last_error}")


def commons_images(queries, limit=6):
    """Открытые фото с Wikimedia Commons (без ключа)."""
    found = []
    for query in queries:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": 8,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 1080,
            },
            headers=UA,
            timeout=60,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "").lower()
            info = (page.get("imageinfo") or [{}])[0]
            thumb = info.get("thumburl")
            if not thumb or not title.endswith((".jpg", ".jpeg", ".png")):
                continue
            meta = info.get("extmetadata", {})
            artist = meta.get("Artist", {}).get("value", "автор неизвестен")
            artist = re.sub(r"<[^>]+>", "", artist).strip()
            license_name = meta.get("LicenseShortName", {}).get("value", "")
            found.append({"url": thumb, "credit": f"{artist} ({license_name})"})
            if len(found) >= limit:
                return found
    return found


def download(url, path):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)


async def speak(text, path):
    await edge_tts.Communicate(text, VOICE, rate="+5%").save(str(path))


def duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]
    )
    return float(out)


def render_slide(image_path, text, out_png):
    base = ImageOps.fit(Image.open(image_path).convert("RGB"), (W, H))
    base = base.convert("RGBA")
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([0, H - 720, W, H], fill=(0, 0, 0, 170))
    base = Image.alpha_composite(base, shade)
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(FONT_BOLD, 60)
    y = H - 660
    for line in textwrap.wrap(text, width=22)[:7]:
        draw.text((60, y), line, font=font, fill=(255, 255, 255, 255))
        y += 84
    base.convert("RGB").save(out_png)


def slide_clip(png, seconds, out_mp4):
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", str(png),
         "-t", f"{seconds:.2f}", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(out_mp4)],
        check=True, capture_output=True,
    )


def render_thumbnail(title, image_path, out_png):
    base = ImageOps.fit(Image.open(image_path).convert("RGB"), (1280, 720)).convert("RGBA")
    shade = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([0, 420, 1280, 720], fill=(0, 0, 0, 170))
    im = Image.alpha_composite(base, shade).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT_BOLD, 74)
    y = 450
    for line in textwrap.wrap(title, width=22)[:3]:
        draw.text((50, y), line, font=font, fill=(255, 255, 255))
        y += 90
    im.save(out_png)


def youtube_client():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=[
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube",
        ],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(video, thumb, meta, credits):
    description = (
        f"{meta['description']}\n\n"
        "Данные о составе: USDA FoodData Central (fdc.nal.usda.gov).\n"
        f"Фото: {'; '.join(credits)} (Wikimedia Commons).\n\n"
        "Общая информация, не медицинская консультация.\n#Shorts"
    )
    body = {
        "snippet": {
            "title": meta["title"][:95],
            "description": description,
            "tags": meta.get("tags", [])[:15],
            "categoryId": "27",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    yt = youtube_client()
    request = yt.videos().insert(
        part="snippet,status",
        body=body,
        media_body=MediaFileUpload(str(video), chunksize=-1, resumable=True),
    )
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response["id"]
    yt.thumbnails().set(
        videoId=video_id, media_body=MediaFileUpload(str(thumb))
    ).execute()
    return video_id


def main():
    OUT.mkdir(exist_ok=True)
    topics, topic = load_topic()
    name = topic["name_ru"]
    print(f"Тема: {name}")

    food_name, fdc_id, nutrients = usda(topic["query_en"])
    facts = "\n".join(f"{k}: {v}" for k, v in nutrients.items())
    data = ask_llm(
        f"Продукт: {name}. Данные USDA ({food_name}, FDC {fdc_id}, на 100 г):\n{facts}\n"
        "Напиши сценарий."
    )

    images = commons_images(data["image_queries"] + [topic["query_en"]])
    if len(images) < 3:
        raise SystemExit("На Wikimedia Commons найдено меньше трёх подходящих фото")

    work = OUT / str(topic["id"])
    work.mkdir(parents=True, exist_ok=True)
    local_images = []
    for i, img in enumerate(images):
        path = work / f"img{i}.jpg"
        download(img["url"], path)
        local_images.append(path)

    script = data["script"].strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()]
    voice = work / "voice.mp3"
    asyncio.run(speak(script, voice))
    total = duration(voice) + 0.5
    if total > 58:
        raise SystemExit(f"Сценарий слишком длинный: {total:.0f} с (нужно до 60 с)")

    weights = [len(s) for s in sentences]
    weight_sum = sum(weights)
    clips = []
    for i, sentence in enumerate(sentences):
        seconds = total * weights[i] / weight_sum
        png = work / f"slide{i}.png"
        render_slide(local_images[i % len(local_images)], sentence, png)
        clip = work / f"clip{i}.mp4"
        slide_clip(png, seconds, clip)
        clips.append(clip)

    (work / "list.txt").write_text(
        "\n".join(f"file '{c.name}'" for c in clips), encoding="utf-8"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
         "-c", "copy", "video.mp4"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", "video.mp4", "-i", "voice.mp3",
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
         "-shortest", "final.mp4"],
        cwd=work, check=True, capture_output=True,
    )

    thumb = work / "thumb.png"
    render_thumbnail(data["title"], local_images[0], thumb)

    credits = [img["credit"] for img in images[: len(local_images)]]
    video_id = upload(work / "final.mp4", thumb, data, credits)
    print(f"Загружено: https://youtu.be/{video_id} (доступ: {PRIVACY})")

    topic["status"] = "done"
    topic["youtube_id"] = video_id
    save_topics(topics)


if __name__ == "__main__":
    main()
