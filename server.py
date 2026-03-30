from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx
import os
import tempfile
import json
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key not configured")
    
    content = await file.read()
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            files={"file": (file.filename, content, file.content_type)},
            data={
                "model": "whisper-1",
                "language": "he",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment"
            }
        )
    
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    
    return response.json()

@app.post("/summarize")
async def summarize(request: dict):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key not configured")
    
    transcript = request.get("transcript", "")
    screen_text = request.get("screen_text", "")
    segments = request.get("segments", [])
    shots_meta = request.get("shots_meta", [])
    
    # Build synced context
    ctx = ""
    if segments:
        ctx = "=== תמלול מסונכרן עם צילומי מסך ===\n"
        for seg in segments:
            t = seg.get("start", 0)
            tl = f"{int(t//60)}:{int(t%60):02d}"
            ctx += f"[{tl}] {seg.get('text','')}\n"
            near = [s for s in shots_meta if abs(s.get("timestamp",0) - t) < 15]
            if near:
                ctx += f"  [📸 צילום: {near[0].get('reason','')}]\n"
    else:
        ctx = f"=== מה המרצה אמר ===\n{transcript}"
    
    if screen_text:
        ctx += f"\n\n=== מה היה על הלוח/מסך ===\n{screen_text}"
    
    system_prompt = """אתה מומחה לסיכום שיעורים אקדמיים לסטודנטים ישראלים.

כללי ברזל:
1. סכם רק את מה שמופיע בחומר
2. נוסחאות תמיד בLaTeX: inline $...$ ושורה נפרדת $$...$$
3. תרגילים — כל שלב בנפרד עם נוסחה מלאה
4. הצלבה — לכל נוסחה ציין מקור: screen/voice/both
5. הסבר כל מושג בעברית פשוטה שכולם יבינו

החזר JSON בלבד:
{
  "title": "נושא השיעור",
  "topics": ["נושא1", "נושא2"],
  "sections": [
    {"type": "explanation", "title": "כותרת", "content": "טקסט"},
    {"type": "formula", "title": "נוסחאות", "formulas": [{"name": "שם", "latex": "$$...$$", "meaning": "הסבר", "source": "screen"}]},
    {"type": "exercise", "title": "תרגיל", "problem": "שאלה", "steps": [{"step": 1, "desc": "מה עושים", "formula": ""}], "answer": "תשובה"},
    {"type": "highlight", "title": "חשוב למבחן", "points": ["נקודה"]}
  ],
  "exercises": [
    {"question": "שאלת תרגול עם LaTeX", "hint": "רמז"}
  ]
}"""
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "max_tokens": 3500,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"סכם:\n\n{ctx}"}
                ]
            }
        )
    
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    
    raw = response.json()["choices"][0]["message"]["content"]
    
    import re
    m = re.search(r'\{[\s\S]*\}', raw)
    try:
        return json.loads(m.group(0) if m else raw)
    except:
        raise HTTPException(500, "Failed to parse summary JSON")

@app.post("/ocr")
async def ocr(request: dict):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key not configured")
    
    images = request.get("images", [])
    if not images:
        return {"text": ""}
    
    content = [{"type": "text", "text": "חלץ את כל מה שמופיע בצילומי המסך: טקסט, נוסחאות בLaTeX, גרפים, טבלאות."}]
    for img in images[-6:]:
        content.append({"type": "image_url", "image_url": {"url": img, "detail": "high"}})
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "max_tokens": 2000, "messages": [{"role": "user", "content": content}]}
        )
    
    if response.status_code != 200:
        return {"text": ""}
    
    return {"text": response.json()["choices"][0]["message"]["content"]}

@app.post("/generate-exercises")
async def generate_exercises(request: dict):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key not configured")
    
    summary = request.get("summary", {})
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "max_tokens": 1500,
                "messages": [{
                    "role": "user",
                    "content": f"על בסיס הסיכום הזה, צור 5 שאלות תרגול ברמת מבחן עם רמזים. נוסחאות בLaTeX. החזר JSON: {{\"exercises\": [{{\"question\": \"...\", \"hint\": \"...\"}}]}}\n\nסיכום: {json.dumps(summary, ensure_ascii=False)}"
                }]
            }
        )
    
    if response.status_code != 200:
        raise HTTPException(500, "Failed")
    
    raw = response.json()["choices"][0]["message"]["content"]
    import re
    m = re.search(r'\{[\s\S]*\}', raw)
    try:
        return json.loads(m.group(0) if m else raw)
    except:
        raise HTTPException(500, "Parse error")
