import './globals.css'

export const metadata = {
  title: 'Foley Agent UI',
  description: 'WebSocket streaming UI for agentic video-to-foley generation'
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
