# Foley Agent Web UI (Next.js)

## Run

1. Start model services first (your existing backend stack).
2. Start WebSocket bridge:
   `./server/run_agent_ws_api.sh`
3. Start frontend:
   `cd web && npm run dev`
4. Open `http://localhost:3000`

## Environment

Optional frontend env (`web/.env.local`):

`NEXT_PUBLIC_AGENT_WS_URL=ws://localhost:8010/ws/foley`

## Notes

- This app is JavaScript-only (no TypeScript).
- Uses Tailwind CSS + shadcn-style structure (`components/ui`).
- The socket expects backend-accessible `video_path` values.
