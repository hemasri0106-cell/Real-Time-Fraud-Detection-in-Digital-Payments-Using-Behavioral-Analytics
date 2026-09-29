import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FingerprintFooter from '../components/FingerprintFooter.jsx'
import { getDeviceId } from '../lib/device.js'
import './Login.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function Login() {
  const navigate = useNavigate()
  const [deviceId] = useState(() => getDeviceId())
  const [email, setEmail] = useState('')
  const [otp, setOtp] = useState('')
  const [step, setStep] = useState('email')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [resendIn, setResendIn] = useState(0)

  useEffect(() => {
    if (resendIn <= 0) return undefined
    const timer = window.setTimeout(() => setResendIn((seconds) => Math.max(0, seconds - 1)), 1000)
    return () => window.clearTimeout(timer)
  }, [resendIn])

  async function post(path, body) {
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-device-id': deviceId },
      body: JSON.stringify(body),
    })
    const data = await response.json().catch(() => null)
    if (!response.ok) throw new Error(data?.detail || `Request failed (${response.status})`)
    return data
  }

  async function sendCode(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const result = await post('/api/auth/send-otp', { email })
      setStep('otp')
      setResendIn(result.resend_after_seconds || 60)
      setOtp('')
    } catch (failure) {
      setError(failure.message || 'Could not send a verification code.')
    } finally {
      setBusy(false)
    }
  }

  async function verifyCode(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const result = await post('/api/auth/verify-otp', { email, otp })
      localStorage.setItem('fraudec_token', result.access_token)
      localStorage.setItem('fraudec_user', JSON.stringify(result.user))
      navigate('/dashboard', { replace: true })
    } catch (failure) {
      setError(failure.message || 'The code could not be verified.')
    } finally {
      setBusy(false)
    }
  }

  function changeEmail() {
    setStep('email')
    setOtp('')
    setError('')
    setResendIn(0)
  }

  return (
    <div className="login">
      <aside className="login__left" aria-label="About FraudEC">
        <div className="login__brand"><span className="login__brand-mark">FRAUDEC</span></div>
        <div className="login__story">
          <div className="login__eyebrow">BEHAVIORAL FRAUD INTELLIGENCE</div>
          <h1>See the signal<br />behind every <span>payment.</span></h1>
          <p>Explore real transaction behavior and historical dataset labels in a focused security workspace.</p>
          <div className="login__visual"><span className="login__visual-mark" aria-hidden="true">⌁</span><span className="login__visual-copy"><strong>Secure email sign-in</strong><span>One-time codes expire after five minutes.</span></span></div>
        </div>
        <FingerprintFooter deviceId={deviceId} sessionLabel="unauthenticated" />
      </aside>

      <main className="login__right">
        <div className="login__mobile-brand" aria-label="FraudEC">FRAUDEC</div>
        <div className="login__form-wrap">
          <div className="login__eyebrow">SECURE ACCESS</div>
          <h1 className="login__title">{step === 'email' ? 'Sign in to FraudEC' : 'Verify your email'}</h1>
          <p className="login__subtitle">{step === 'email' ? 'We’ll send a one-time verification code to your email address.' : <>Enter the six-digit code sent to <strong className="login__verified-email">{email}</strong>.</>}</p>

          {step === 'email' ? (
            <form className="login__form" onSubmit={sendCode}>
              <label className="login__field" htmlFor="email"><span className="login__field-label">Email address</span><input id="email" className="login__input" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required maxLength={254} /></label>
              <button className="login__submit" type="submit" disabled={busy}>{busy ? 'Sending code…' : 'Send OTP'}</button>
            </form>
          ) : (
            <form className="login__form" onSubmit={verifyCode}>
              <label className="login__field" htmlFor="otp"><span className="login__field-label">Six-digit code</span><input id="otp" className="login__input login__otp-input" type="text" inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={otp} onChange={(event) => setOtp(event.target.value.replace(/\D/g, '').slice(0, 6))} required aria-describedby="otp-help" /></label>
              <span id="otp-help" className="login__hint">The code expires after five minutes.</span>
              <button className="login__submit" type="submit" disabled={busy || otp.length !== 6}>{busy ? 'Verifying…' : 'Verify OTP'}</button>
              <div className="login__otp-actions"><button type="button" className="login__text-button" onClick={changeEmail} disabled={busy}>Change email</button><button type="button" className="login__text-button" onClick={() => sendCode({ preventDefault() {} })} disabled={busy || resendIn > 0}>{resendIn > 0 ? `Resend in ${resendIn}s` : 'Resend code'}</button></div>
            </form>
          )}
          <div className="login__error" role="alert" aria-live="polite">{error}</div>
          <div className="login__security-note"><span aria-hidden="true">◈</span> Verification codes are single-use and never stored in plaintext.</div>
          <FingerprintFooter deviceId={deviceId} sessionLabel="unauthenticated" />
        </div>
      </main>
    </div>
  )
}
