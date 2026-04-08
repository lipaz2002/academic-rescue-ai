from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
import os
import json
import re
import time
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ─── הגנות ─────────────────────────────────────────────────────
MAX_FILE_MB = 25
RATE_PER_HOUR = 10
RATE_PER_DAY = 30
MAX_TRANSCRIPT = 50000

rate_store = defaultdict(list)

def get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else request.client.host

def check_rate(ip: str):
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if t > now - 86400]
    last_hour = [t for t in rate_store[ip] if t > now - 3600]
    if len(last_hour) >= RATE_PER_HOUR:
        mins = int((min(last_hour) + 3600 - now) / 60)
        raise HTTPException(429, f"יותר מדי בקשות. נסה שוב בעוד {mins} דקות.")
    if len(rate_store[ip]) >= RATE_PER_DAY:
        raise HTTPException(429, f"הגעת למכסה היומית ({RATE_PER_DAY} בקשות). נסה מחר.")
    rate_store[ip].append(now)

# ─── Static ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/static/sw.js")
async def sw():
    return FileResponse("static/sw.js", media_type="application/javascript")

@app.get("/static/manifest.json")
async def manifest():
    return FileResponse("static/manifest.json", media_type="application/json")

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

@app.get("/usage")
async def usage(request: Request):
    ip = get_ip(request)
    now = time.time()
    reqs = [t for t in rate_store.get(ip, []) if t > now - 86400]
    return {
        "last_hour": len([t for t in reqs if t > now - 3600]),
        "last_day": len(reqs),
        "limit_hour": RATE_PER_HOUR,
        "limit_day": RATE_PER_DAY,
    }

# ─── Transcribe ─────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(request))

    content = await file.read()

    if len(content) / 1024 / 1024 > MAX_FILE_MB:
        raise HTTPException(413, f"קובץ גדול מדי. מקסימום {MAX_FILE_MB}MB.")
    if len(content) < 500:
        raise HTTPException(400, "הקובץ ריק או קצר מדי.")

    ext = (file.filename or "").lower().split(".")[-1]
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct.startswith("video/") or
            ext in ["mp3","mp4","wav","m4a","webm","ogg","flac","aac"]):
        raise HTTPException(400, f"סוג קובץ לא נתמך.")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                files={"file": (file.filename or "audio.mp3", content, ct or "audio/mpeg")},
                data={"model": "whisper-1", "language": "he",
                      "response_format": "verbose_json",
                      "timestamp_granularities[]": "segment"}
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "פסק זמן — נסה קובץ קצר יותר.")
    except Exception as e:
        raise HTTPException(502, f"שגיאת חיבור: {str(e)[:80]}")

    if r.status_code == 429:
        raise HTTPException(429, "יותר מדי בקשות. נסה שוב בעוד דקה.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"שגיאת תמלול: {r.text[:150]}")

    return r.json()

# ─── Summarize ──────────────────────────────────────────────────
@app.post("/summarize")
async def summarize(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(req))

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    transcript = str(body.get("transcript", ""))[:MAX_TRANSCRIPT]
    screen_text = str(body.get("screen_text", ""))[:5000]
    segments = body.get("segments", [])[:200]
    shots_meta = body.get("shots_meta", [])[:50]

    if len(transcript) < 50:
        raise HTTPException(400, "התמלול קצר מדי לסיכום.")

    ctx = ""
    if segments:
        ctx = "=== תמלול ===\n"
        for seg in segments:
            t = seg.get("start", 0)
            ctx += f"[{int(t//60)}:{int(t%60):02d}] {str(seg.get('text',''))[:500]}\n"
            near = [s for s in shots_meta if abs(s.get("timestamp", 0) - t) < 15]
            if near:
                ctx += f"  [📸 {near[0].get('reason','')}]\n"
    else:
        ctx = f"=== שיעור ===\n{transcript}"
    if screen_text:
        ctx += f"\n\n=== מסך ===\n{screen_text}"

    system = """אתה מומחה לסיכום שיעורים אקדמיים לסטודנטים ישראלים.
כללים: נוסחאות בLaTeX ($...$ ו$$...$$), תרגילים שלב-שלב, עברית פשוטה.
החזר JSON בלבד:
{"title":"נושא","topics":["נושא"],"sections":[
{"type":"explanation","title":"כותרת","content":"טקסט"},
{"type":"formula","title":"נוסחאות","formulas":[{"name":"שם","latex":"$$...$$","meaning":"הסבר","source":"voice"}]},
{"type":"exercise","title":"תרגיל","problem":"שאלה","steps":[{"step":1,"desc":"מה","formula":""}],"answer":"תשובה"},
{"type":"highlight","title":"חשוב למבחן","points":["נקודה"]}
],"exercises":[{"question":"שאלה","hint":"רמז"}]}"""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": 3500,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": f"סכם:\n\n{ctx[:30000]}"}]}
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "הסיכום לקח יותר מדי זמן. נסה שוב.")
    except Exception as e:
        raise HTTPException(502, f"שגיאת חיבור: {str(e)[:80]}")

    if r.status_code == 429:
        raise HTTPException(429, "יותר מדי בקשות. נסה שוב.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"שגיאה: {r.text[:150]}")

    raw = r.json()["choices"][0]["message"]["content"]
    m = re.search(r'\{[\s\S]*\}', raw)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:
        raise HTTPException(500, "שגיאה בפענוח הסיכום. נסה שוב.")

# ─── OCR ────────────────────────────────────────────────────────
@app.post("/ocr")
async def ocr(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(req))

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    images = [img for img in body.get("images", [])[:6]
              if isinstance(img, str) and img.startswith("data:image/")]
    if not images:
        return {"text": ""}

    content = [{"type": "text", "text": "חלץ כל טקסט ונוסחאות (LaTeX) מהצילומים."}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img, "detail": "high"}})

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": 2000,
                      "messages": [{"role": "user", "content": content}]}
            )
        if r.status_code == 200:
            return {"text": r.json()["choices"][0]["message"]["content"]}
    except Exception:
        pass

    return {"text": ""}
