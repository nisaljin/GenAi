'use client'

import React, { useEffect, useRef, useState } from 'react'
import {
  Plus,
  Lightbulb,
  Paperclip,
  Image,
  FileCode,
  Bolt,
  SendHorizontal,
  LoaderCircle,
  Video
} from 'lucide-react'

const REASONING_EVENT_TYPES = new Set([
  'run_started',
  'planning_completed',
  'vlm_keyframes_extracted',
  'vlm_keyframes_downsampled',
  'vlm_request_started',
  'vlm_response_received',
  'vlm_request_failed',
  'attempt_started',
  'clap_scored',
  'decision_made',
  'event_completed',
  'perception_completed',
  'video_prepared'
])

function TypewriterText({ text, animate = false }) {
  const [visibleText, setVisibleText] = useState(animate ? '' : text)

  useEffect(() => {
    if (!animate) {
      setVisibleText(text)
      return
    }

    const tokens = text.split(/(\s+)/).filter(Boolean)
    if (tokens.length === 0) {
      setVisibleText('')
      return
    }

    let index = 0
    const timer = setInterval(() => {
      index += 1
      setVisibleText(tokens.slice(0, index).join(''))
      if (index >= tokens.length) clearInterval(timer)
    }, 35)

    return () => clearInterval(timer)
  }, [text, animate])

  return <div className="whitespace-pre-wrap">{visibleText}</div>
}

function AnnouncementBadge({ text, href = '#' }) {
  const content = (
    <>
      <span
        className="absolute top-0 left-0 right-0 h-1/2 pointer-events-none opacity-70 mix-blend-overlay"
        style={{ background: 'radial-gradient(ellipse at center top, rgba(255, 255, 255, 0.15) 0%, transparent 70%)' }}
      />
      <span
        className="absolute -top-px left-1/2 -translate-x-1/2 h-[2px] w-[100px] opacity-60"
        style={{
          background: 'linear-gradient(90deg, transparent 0%, rgba(37, 119, 255, 0.8) 20%, rgba(126, 93, 225, 0.8) 50%, rgba(59, 130, 246, 0.8) 80%, transparent 100%)',
          filter: 'blur(0.5px)'
        }}
      />
      <Bolt className="size-4 relative z-10 text-white" />
      <span className="relative z-10 text-white font-medium">{text}</span>
    </>
  )

  const className = 'relative inline-flex items-center gap-2 px-5 py-2 min-h-[40px] rounded-full text-sm overflow-hidden transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] cursor-pointer'
  const style = {
    background: 'linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
    backdropFilter: 'blur(20px) saturate(140%)',
    boxShadow: 'inset 0 1px rgba(255,255,255,0.2), inset 0 -1px rgba(0,0,0,0.1), 0 8px 32px -8px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.08)'
  }

  return href !== '#' ? (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className} style={style}>
      {content}
    </a>
  ) : (
    <button className={className} style={style}>{content}</button>
  )
}

function RayBackground() {
  return (
    <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none select-none">
      <div className="absolute inset-0 bg-[#0f0f0f]" />
      <div
        className="absolute left-1/2 -translate-x-1/2 w-[4000px] h-[1800px] sm:w-[6000px]"
        style={{
          background:
            'radial-gradient(circle at center 800px, rgba(20, 136, 252, 0.8) 0%, rgba(20, 136, 252, 0.35) 14%, rgba(20, 136, 252, 0.18) 18%, rgba(20, 136, 252, 0.08) 22%, rgba(17, 17, 20, 0.2) 25%)'
        }}
      />
      <div className="absolute top-[175px] left-1/2 w-[1600px] h-[1600px] sm:top-1/2 sm:w-[3043px] sm:h-[2865px]" style={{ transform: 'translate(-50%) rotate(180deg)' }}>
        <div
          className="absolute w-full h-full rounded-full -mt-[13px]"
          style={{
            background: 'radial-gradient(43.89% 25.74% at 50.02% 97.24%, #111114 0%, #0f0f0f 100%)',
            border: '16px solid white',
            transform: 'rotate(180deg)',
            zIndex: 5
          }}
        />
        <div className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[11px]" style={{ border: '23px solid #b7d7f6', transform: 'rotate(180deg)', zIndex: 4 }} />
        <div className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[8px]" style={{ border: '23px solid #8fc1f2', transform: 'rotate(180deg)', zIndex: 3 }} />
        <div className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[4px]" style={{ border: '23px solid #64acf6', transform: 'rotate(180deg)', zIndex: 2 }} />
        <div
          className="absolute w-full h-full rounded-full bg-[#0f0f0f]"
          style={{ border: '20px solid #1172e2', boxShadow: '0 -15px 24.8px rgba(17, 114, 226, 0.6)', transform: 'rotate(180deg)', zIndex: 1 }}
        />
      </div>
    </div>
  )
}

function EventFeed({ events, isRunning = false }) {
  const scrollRef = useRef(null)
  const contentRef = useRef(null)

  const scrollToBottom = (smooth = false) => {
    if (!scrollRef.current) return
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto'
    })
  }

  useEffect(() => {
    scrollToBottom(true)
    const raf = requestAnimationFrame(() => scrollToBottom(false))
    return () => cancelAnimationFrame(raf)
  }, [events, isRunning])

  useEffect(() => {
    if (!contentRef.current) return

    const observer = new ResizeObserver(() => {
      scrollToBottom(false)
    })
    observer.observe(contentRef.current)

    return () => observer.disconnect()
  }, [])

  return (
    <div ref={scrollRef} className="w-full max-w-[860px] rounded-2xl border border-[#2f3a4b] bg-[#0a1018] p-4 sm:p-5 max-h-[460px] sm:max-h-[520px] overflow-y-auto shadow-[0_12px_40px_rgba(0,0,0,0.45)]">
      <div className="text-xs uppercase tracking-wide text-[#8a8a8f] mb-3">Live Agent Events</div>
      <div ref={contentRef} className="space-y-2">
        {events.length === 0 && <p className="text-sm text-[#6a6a6f]">No events yet.</p>}
        {events.map((evt) => (
          <div key={evt.id} className={`rounded-xl px-3 py-2 text-sm transition-all duration-300 ${evt.role === 'user' ? 'bg-[#19385a] text-blue-100 ml-8 border border-[#2d5278]' : 'bg-[#151a23] text-[#e4e6eb] mr-8 border border-[#293241]'}`}>
            <div className="text-[11px] text-[#8a8a8f] mb-1">{evt.label}</div>
            <TypewriterText text={evt.text} animate={Boolean(evt.animateText)} />
            {evt.media?.url && evt.media?.kind === 'video' && (
              <video
                src={evt.media.url}
                controls
                className="mt-3 w-full max-h-72 rounded-lg border border-white/10 bg-black object-contain"
              />
            )}
            {evt.media?.url && evt.media?.kind === 'audio' && (
              <audio
                src={evt.media.url}
                controls
                className="mt-3 w-full"
              />
            )}
          </div>
        ))}
        {isRunning && (
          <div className="rounded-xl px-3 py-2 text-sm bg-[#151a23] text-[#e4e6eb] mr-8 border border-[#293241]">
            <div className="text-[11px] text-[#8a8a8f] mb-1">Generating</div>
            <div className="inline-flex items-center gap-2">
              <span>Awaiting model response</span>
              <span className="inline-flex gap-1" aria-hidden="true">
                <span className="size-1.5 rounded-full bg-[#8ba6c7] animate-bounce [animation-delay:0ms]" />
                <span className="size-1.5 rounded-full bg-[#8ba6c7] animate-bounce [animation-delay:120ms]" />
                <span className="size-1.5 rounded-full bg-[#8ba6c7] animate-bounce [animation-delay:240ms]" />
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ChatInput({ onSend, disabled = false }) {
  const [message, setMessage] = useState('')
  const [videoFile, setVideoFile] = useState(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [showAttachMenu, setShowAttachMenu] = useState(false)
  const textareaRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [message])

  useEffect(() => {
    if (!videoFile) {
      setVideoPreviewUrl('')
      return
    }

    const url = URL.createObjectURL(videoFile)
    setVideoPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  const handleSubmit = () => {
    const prompt = message.trim()
    if ((!prompt && !videoFile) || disabled) return
    if (onSend) onSend({ prompt, videoFile })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const pickVideo = () => {
    if (fileRef.current) fileRef.current.click()
  }

  return (
    <div className="relative w-full max-w-[860px] mx-auto">
      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files && e.target.files[0]
          setVideoFile(f || null)
        }}
      />

      <div className="absolute -inset-[1px] rounded-2xl bg-gradient-to-b from-white/[0.08] to-transparent pointer-events-none" />
      <div className="relative rounded-2xl bg-[#1e1e22] ring-1 ring-white/[0.08] shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_2px_20px_rgba(0,0,0,0.4)]">
        <div className="px-4 pt-3 text-xs text-[#8a8a8f] flex items-center gap-2">
          <Video className="size-4 text-blue-300" />
          {videoFile ? `Video attached: ${videoFile.name}` : 'No video attached. Prompt-only audio generation will run.'}
        </div>
        {videoPreviewUrl && (
          <div className="px-4 pt-3">
            <video
              src={videoPreviewUrl}
              controls
              muted
              className="w-full max-h-56 rounded-lg border border-white/10 bg-black object-contain"
            />
          </div>
        )}

        {!videoFile ? (
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe the target foley sound."
              className="w-full resize-none bg-transparent text-[15px] text-white placeholder-[#5a5a5f] px-5 pt-4 pb-3 focus:outline-none min-h-[84px] max-h-[200px]"
              style={{ height: '84px' }}
              disabled={disabled}
            />
          </div>
        ) : (
          <div className="px-5 pt-4 pb-3 text-sm text-[#a8b3c2]">
            Prompt is skipped in video mode. The agent will use visual perception from uploaded frames.
          </div>
        )}

        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-1">
            <div className="relative">
              <button
                onClick={() => setShowAttachMenu(!showAttachMenu)}
                className="flex items-center justify-center size-8 rounded-full bg-white/[0.08] hover:bg-white/[0.12] text-[#8a8a8f] hover:text-white transition-all duration-200 active:scale-95"
                disabled={disabled}
              >
                <Plus className={`size-4 transition-transform duration-200 ${showAttachMenu ? 'rotate-45' : ''}`} />
              </button>

              {showAttachMenu && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowAttachMenu(false)} />
                  <div className="absolute bottom-full left-0 mb-2 z-50 bg-[#1a1a1e]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl shadow-black/50 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200">
                    <div className="p-1.5 min-w-[220px]">
                      <button
                        onClick={() => {
                          pickVideo()
                          setShowAttachMenu(false)
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[#a0a0a5] hover:bg-white/5 hover:text-white transition-all duration-150"
                      >
                        <Paperclip className="size-4" />
                        <span className="text-sm">Upload video</span>
                      </button>
                      <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[#a0a0a5] hover:bg-white/5 hover:text-white transition-all duration-150">
                        <Image className="size-4" />
                        <span className="text-sm">Add image reference</span>
                      </button>
                      <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[#a0a0a5] hover:bg-white/5 hover:text-white transition-all duration-150">
                        <FileCode className="size-4" />
                        <span className="text-sm">Import scene metadata</span>
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <button className="flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium text-[#6a6a6f] hover:text-white hover:bg-white/5 transition-all duration-200">
              <Lightbulb className="size-4" />
              <span className="hidden sm:inline">Plan</span>
            </button>

            <button
              onClick={handleSubmit}
              disabled={(!message.trim() && !videoFile) || disabled}
              className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-[#1488fc] hover:bg-[#1a94ff] text-white transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 shadow-[0_0_20px_rgba(20,136,252,0.3)]"
            >
              {disabled ? <LoaderCircle className="size-4 animate-spin" /> : <SendHorizontal className="size-4" />}
              <span className="hidden sm:inline">Generate</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function mapEventToDisplay(event) {
  const type = event.type
  const p = event.payload || {}
  const formatVlmLog = (vlmLog = '') => {
    const lines = String(vlmLog)
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
    if (lines.length === 0) return 'No VLM timeline lines were returned.'
    return `Timeline\n${lines.map((line) => `• ${line}`).join('\n')}`
  }
  const formatPlanEvents = (events = []) => {
    if (!Array.isArray(events) || events.length === 0) return 'No plan events were generated.'
    return events
      .map((item, i) => {
        const ts = Number(item.timestamp_sec ?? 0).toFixed(2)
        const dur = Number(item.duration_sec ?? 0).toFixed(2)
        const prompt = String(item.original_prompt || '').trim()
        return `${i + 1}. [${ts}s +${dur}s] ${prompt}`
      })
      .join('\n')
  }

  if (type === 'run_started') {
    return {
      eventType: type,
      label: 'Run Started',
      text: p.mode === 'audio_only'
        ? `Prompt-only mode\nPrompt: ${p.prompt || ''}`
        : `Video mode\nVideo: ${p.video_path || ''}\nMax retries: ${p.max_retries}\nCLAP threshold: ${p.clap_threshold}`
    }
  }
  if (type === 'vlm_keyframes_extracted') {
    return {
      eventType: type,
      label: 'VLM Keyframes',
      text: `Extracted ${p.keyframe_count} keyframes (diff threshold ${p.threshold}).`
    }
  }
  if (type === 'vlm_keyframes_downsampled') {
    return {
      eventType: type,
      label: 'VLM Sampling',
      text: `Downsampled keyframes ${p.before} -> ${p.after} (MAX_PERCEPTION_FRAMES=${p.max_frames}).`
    }
  }
  if (type === 'vlm_request_started') {
    const timestamps = Array.isArray(p.frame_timestamps) ? p.frame_timestamps.join(', ') : ''
    return {
      eventType: type,
      label: 'VLM Request',
      text: `Model: ${p.model}\nFrames: ${p.frame_count}\nTimestamps: ${timestamps}\nEndpoint: ${p.endpoint}`
    }
  }
  if (type === 'vlm_response_received') {
    return {
      eventType: type,
      label: 'VLM Response',
      text: `Received VLM output (${p.log_length} chars).\nPreview:\n${p.preview || ''}`
    }
  }
  if (type === 'vlm_request_failed') return { eventType: type, label: 'VLM Failed', text: p.error || 'Unknown VLM failure' }
  if (type === 'planning_completed') {
    return {
      eventType: type,
      label: 'Planning',
      text: `Planned ${p.event_count} event(s).\n${formatPlanEvents(p.events)}`
    }
  }
  if (type === 'attempt_started') return { eventType: type, label: `Attempt ${p.attempt}`, text: `Prompt: ${p.prompt}` }
  if (type === 'clap_scored') return { eventType: type, label: 'CLAP Score', text: `Score ${p.score} (threshold ${p.threshold})` }
  if (type === 'decision_made') return { eventType: type, label: `Decision: ${p.action}`, text: `${p.reasoning}\n(confidence ${p.confidence})` }
  if (type === 'event_completed') return { eventType: type, label: 'Event Completed', text: `Final score ${p.final_score}` }
  if (type === 'run_completed') {
    const output = p.output_video_path || p.output_audio_path || ''
    const mediaKind = p.output_video_path ? 'video' : (p.output_audio_path ? 'audio' : '')
    return {
      eventType: type,
      label: 'Run Completed',
      text: `Output: ${output}\nTrace: ${p.report_path || ''}`,
      link: '',
      media: p.output_url && mediaKind ? { kind: mediaKind, url: p.output_url } : null
    }
  }
  if (type === 'run_failed') return { eventType: type, label: 'Run Failed', text: p.error || 'Unknown error' }
  if (type === 'video_prepared') return { eventType: type, label: 'Video Prepared', text: `Duration: ${p.duration_sec}s | Trimmed: ${String(p.was_trimmed)}` }
  if (type === 'perception_completed') return { eventType: type, label: 'Perception Timeline', text: formatVlmLog(p.vlm_log || '') }
  return { eventType: type, label: type, text: JSON.stringify(p), link: '', media: null }
}

export function BoltStyleChat({
  title = 'What will you',
  subtitle = 'Upload a video or skip upload for prompt-only audio. Generate and watch the agent reason live.',
  announcementText = 'Foley Agent Live Streaming',
  announcementHref = '#',
  websocketUrl = process.env.NEXT_PUBLIC_AGENT_WS_URL || 'ws://localhost:8010/ws/foley',
  apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8010'
}) {
  const [events, setEvents] = useState([])
  const [isRunning, setIsRunning] = useState(false)
  const [hasSubmitted, setHasSubmitted] = useState(false)
  const [lastRequest, setLastRequest] = useState(null)
  const [canRetry, setCanRetry] = useState(false)
  const wsRef = useRef(null)
  const eventQueueRef = useRef([])
  const queueTimerRef = useRef(null)
  const queueBusyRef = useRef(false)

  const flushNextEvent = () => {
    if (eventQueueRef.current.length === 0) {
      queueBusyRef.current = false
      return
    }

    queueBusyRef.current = true
    const next = eventQueueRef.current.shift()
    setEvents((prev) => [...prev, next])
    queueTimerRef.current = setTimeout(flushNextEvent, next.animateText ? 240 : 120)
  }

  const enqueueEvent = (eventObj) => {
    eventQueueRef.current.push(eventObj)
    if (!queueBusyRef.current) flushNextEvent()
  }

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (queueTimerRef.current) clearTimeout(queueTimerRef.current)
    }
  }, [])

  const pushEvent = (role, label, text, link = '', animateText = false) => {
    enqueueEvent({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      role,
      label,
      text,
      link,
      media: null,
      animateText
    })
  }

  const handleGenerate = async ({ prompt, videoFile }) => {
    if (isRunning) return

    setLastRequest({ prompt, videoFile: videoFile || null })
    setCanRetry(false)
    setHasSubmitted(true)
    setIsRunning(true)
    setEvents([])
    eventQueueRef.current = []
    queueBusyRef.current = false
    if (queueTimerRef.current) clearTimeout(queueTimerRef.current)
    pushEvent('user', 'Generation Request', `Prompt: ${prompt}\nVideo: ${videoFile ? videoFile.name : 'none (prompt-only)'}`)

    let uploadedVideoPath = ''
    if (videoFile) {
      try {
        pushEvent('assistant', 'Upload', `Uploading ${videoFile.name}...`)
        const uploadUrl = `${apiBaseUrl}/upload-video?filename=${encodeURIComponent(videoFile.name)}`
        const uploadRes = await fetch(uploadUrl, {
          method: 'PUT',
          headers: {
            'Content-Type': videoFile.type || 'application/octet-stream'
          },
          body: videoFile
        })

        if (!uploadRes.ok) {
          const bodyText = await uploadRes.text()
          throw new Error(`Upload failed: ${bodyText}`)
        }

        const uploaded = await uploadRes.json()
        uploadedVideoPath = uploaded.video_path || ''
        pushEvent('assistant', 'Upload Complete', `Stored at backend path: ${uploadedVideoPath}`)
      } catch (err) {
        pushEvent('assistant', 'Upload Failed', String(err))
        setIsRunning(false)
        setCanRetry(true)
        return
      }
    }

    const ws = new WebSocket(websocketUrl)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          action: 'start',
          prompt,
          video_path: uploadedVideoPath
        })
      )
    }

    ws.onmessage = (messageEvent) => {
      try {
        const event = JSON.parse(messageEvent.data)
        const display = mapEventToDisplay(event)
        enqueueEvent({
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          role: 'assistant',
          label: display.label,
          text: display.text,
          link: display.link || '',
          media: display.media || null,
          animateText: REASONING_EVENT_TYPES.has(display.eventType)
        })

        if (event.type === 'run_completed' || event.type === 'run_failed') {
          setIsRunning(false)
          setCanRetry(event.type === 'run_failed')
          ws.close()
        }
      } catch (err) {
        pushEvent('assistant', 'Parse Error', String(err))
        setCanRetry(true)
      }
    }

    ws.onerror = () => {
      pushEvent('assistant', 'WebSocket Error', 'Failed to connect to the Foley agent stream.')
      setIsRunning(false)
      setCanRetry(true)
    }

    ws.onclose = () => {
      setIsRunning(false)
    }
  }

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen w-full overflow-hidden bg-[#0f0f0f]">
      <RayBackground />

      <div className="absolute top-[70px]">
        <AnnouncementBadge text={announcementText} href={announcementHref} />
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full px-4 py-16 sm:py-20">
        <div className="text-center mb-6">
          <h1 className="text-4xl sm:text-5xl font-bold text-white tracking-tight mb-1">
            {title}{' '}
            <span className="bg-gradient-to-b from-[#4da5fc] via-[#4da5fc] to-white bg-clip-text text-transparent italic">score</span>{' '}
            audio?
          </h1>
          <p className="text-base font-semibold sm:text-lg text-[#8a8a8f] max-w-2xl">{subtitle}</p>
        </div>

        {!hasSubmitted && (
          <div className="w-full max-w-[860px] mb-6 mt-2">
            <ChatInput onSend={handleGenerate} disabled={isRunning} />
          </div>
        )}

        {hasSubmitted && (
          <div className="w-full max-w-[860px] mb-6 sm:mb-8">
            <EventFeed events={events} isRunning={isRunning} />
          </div>
        )}

        {hasSubmitted && !isRunning && (
          <div className="flex items-center gap-3">
            {canRetry && lastRequest && (
              <button
                onClick={() => handleGenerate(lastRequest)}
                className="px-4 py-2 rounded-full text-sm text-white bg-[#1666b4] hover:bg-[#1b75cd] transition-colors"
              >
                Retry
              </button>
            )}
            <button
              onClick={() => {
                setHasSubmitted(false)
                setEvents([])
                setCanRetry(false)
                eventQueueRef.current = []
                queueBusyRef.current = false
                if (queueTimerRef.current) clearTimeout(queueTimerRef.current)
              }}
              className="px-4 py-2 rounded-full text-sm text-white bg-white/10 hover:bg-white/20 transition-colors"
            >
              New generation
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
