import { useEffect, useState } from 'react'
import './FingerprintFooter.css'

function utcTimestamp() {
  return `${new Date().toISOString().slice(0, 19)}Z`
}

export default function FingerprintFooter({ deviceId, sessionLabel }) {
  const [time, setTime] = useState(utcTimestamp())

  useEffect(() => {
    const interval = setInterval(() => setTime(utcTimestamp()), 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="fingerprint">
      <div className="fingerprint__row">
        <span className="fingerprint__label">DEVICE</span>
        <span className="fingerprint__value" title={deviceId}>
          {deviceId}
        </span>
      </div>
      <div className="fingerprint__row">
        <span className="fingerprint__label">SESSION</span>
        <span
          className={`fingerprint__value ${
            sessionLabel === 'unauthenticated' ? 'fingerprint__value--muted' : 'fingerprint__value--signal'
          }`}
        >
          {sessionLabel}
        </span>
      </div>
      <div className="fingerprint__row">
        <span className="fingerprint__label">TIME</span>
        <span className="fingerprint__value">{time}</span>
      </div>
    </div>
  )
}
