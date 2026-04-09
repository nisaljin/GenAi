# Foley Agent Web UI (Next.js)

Frontend for the agentic Foley pipeline.

It provides:
- prompt-only audio generation flow
- video upload flow
- live websocket event timeline
- animated reasoning/event rendering
- inline output playback (`audio`/`video`)

## 1. Prerequisites

- Node.js 18+
- backend services running:
  - model API (`:8000`)
  - WS bridge (`:8010`)

## 2. Install and Run

```bash
cd web
npm install
npm run dev
```

Open: `http://localhost:3000`

## 3. Environment

Create `web/.env.local` as needed:

```env
NEXT_PUBLIC_AGENT_WS_URL=ws://localhost:8010/ws/foley
NEXT_PUBLIC_AGENT_API_URL=http://localhost:8010
```

Defaults in code:
- WS URL defaults to `ws://localhost:8010/ws/foley`
- API URL defaults to `http://localhost:8010`

## 4. UX Behavior

### 4.1 Input behavior

- No video attached: prompt textarea is shown.
- Video attached: prompt box is hidden (video mode uses perception flow).
- Local video preview appears inline before submit.

### 4.2 Event feed

- Opaque cards for readability.
- Sequential queue so events appear gracefully.
- Auto-scroll pinned to latest event.
- Typewriter-style animation for reasoning events.
- “Generating” loading card while awaiting model responses.

### 4.3 Retry

- If run fails, `Retry` button appears.
- Retry reuses last request payload.

### 4.4 Output playback

- `run_completed` with audio renders inline `<audio controls>`.
- `run_completed` with video renders inline `<video controls>`.

## 5. Event Mapping

Event mapping lives in:
- `components/ui/bolt-style-chat.jsx`

When backend event schema changes, update:
- `mapEventToDisplay(...)`
- reasoning animation event-type set
- media rendering fields

## 6. Troubleshooting

### Upload fails (`TypeError: Failed to fetch`)

Likely backend CORS/preflight issue (`OPTIONS /upload-video` failing). Check WS bridge logs and CORS config.

### WebSocket error card appears immediately

Check:
- WS bridge is running on expected port
- URL in `.env.local` matches backend
- websocket dependencies installed in backend runtime

### Media doesn’t play in feed

Check:
- artifact URL returned in `run_completed.payload.output_url`
- backend can serve file from `/artifacts/...`
- browser console for media/network errors

## 7. Tech Notes

- JavaScript-only frontend (no TypeScript in this project).
- Tailwind-based styling and component structure under `components/ui`.
