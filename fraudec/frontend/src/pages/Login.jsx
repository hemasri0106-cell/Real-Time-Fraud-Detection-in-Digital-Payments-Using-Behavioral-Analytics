import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FingerprintFooter from '../components/FingerprintFooter.jsx'
import { getDeviceId } from '../lib/device.js'
import './Login.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function Login() {
  const navigate = useNavigate()
  const [deviceId] = useState(() => getDeviceId())
  const [mode, setMode] = useState('signin')
  const [error, setError] = useState('')
  const [sessionLabel, setSessionLabel] = useState('unauthenticated')
  const [submitting, setSubmitting] = useState(false)

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const [suUsername, setSuUsername] = useState('')
  const [suEmail, setSuEmail] = useState('')
  const [suPassword, setSuPassword] = useState('')
  const [suConfirm, setSuConfirm] = useState('')

  function switchMode(nextMode) {
    setMode(nextMode)
    setError('')
  }

  async function loginRequest(loginUsername, loginPassword) {
    const body = new URLSearchParams({ username: loginUsername, password: loginPassword })
    const response = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'x-device-id': deviceId,
      },
      body,
    })
    const data = await response.json().catch(() => null)
    return { response, data }
  }

  function completeAuth(data) {
    localStorage.setItem('fraudec_token', data.access_token)
    localStorage.setItem('fraudec_user', JSON.stringify(data.user))
    setSessionLabel(data.user?.username || 'authenticated')
    navigate('/dashboard')
  }

  async function handleSignIn(event) {
    event.preventDefault()
    setError('')
    setSubmitting(true)

    try {
      const { response, data } = await loginRequest(username, password)

      if (!response.ok) {
        const detail = data && typeof data.detail === 'string' ? data.detail : null
        setError(detail || 'Sign-in failed. Check your credentials and try again.')
        setSubmitting(false)
        return
      }

      completeAuth(data)
    } catch {
      setError('Could not reach the server. Is the backend running?')
      setSubmitting(false)
    }
  }

  async function handleSignUp(event) {
    event.preventDefault()
    setError('')

    if (suPassword.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }
    if (suPassword !== suConfirm) {
      setError('Passwords do not match')
      return
    }

    setSubmitting(true)

    try {
      const registerResponse = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: suUsername,
          email: suEmail,
          password: suPassword,
          role: 'analyst',
        }),
      })

      if (registerResponse.status === 201) {
        const { response: loginResponse, data: loginData } = await loginRequest(suUsername, suPassword)
        if (!loginResponse.ok) {
          setError('Account created. Please sign in.')
          setSubmitting(false)
          switchMode('signin')
          return
        }
        completeAuth(loginData)
        return
      }

      if (registerResponse.status === 400) {
        const data = await registerResponse.json().catch(() => null)
        const detail = data && typeof data.detail === 'string' ? data.detail : null
        setError(detail || 'Username or email already registered')
        setSubmitting(false)
        return
      }

      setError('Could not create account')
      setSubmitting(false)
    } catch {
      setError('Could not reach the server. Is the backend running?')
      setSubmitting(false)
    }
  }

  const isSignIn = mode === 'signin'

  return (
    <div className="login">
      <div className="login__left">
        <div className="login__brand">
          <span className="login__brand-mark">FRAUDEC</span>
          <p className="login__tagline">Real-time fraud signal, watched continuously.</p>
        </div>

      </div>

      <div className="login__right">
        <div className="login__form-wrap">
          <div className="login__tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={isSignIn}
              className={`login__tab ${isSignIn ? 'login__tab--active' : ''}`}
              onClick={() => switchMode('signin')}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={!isSignIn}
              className={`login__tab ${!isSignIn ? 'login__tab--active' : ''}`}
              onClick={() => switchMode('signup')}
            >
              Create account
            </button>
          </div>

          <h1 className="login__title">{isSignIn ? 'Sign in' : 'Create account'}</h1>
          <p className="login__subtitle">
            {isSignIn ? 'Analyst and admin access only' : 'Register for analyst access'}
          </p>

          {isSignIn ? (
            <form className="login__form" onSubmit={handleSignIn}>
              <label className="login__field">
                <span className="login__field-label">Username</span>
                <input
                  className="login__input"
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </label>

              <label className="login__field">
                <span className="login__field-label">Password</span>
                <input
                  className="login__input"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </label>

              <button className="login__submit" type="submit" disabled={submitting}>
                {submitting ? 'Signing in…' : 'Sign in'}
              </button>

              <div className="login__error" role="alert">
                {error}
              </div>
            </form>
          ) : (
            <form className="login__form" onSubmit={handleSignUp}>
              <label className="login__field">
                <span className="login__field-label">Username</span>
                <input
                  className="login__input"
                  type="text"
                  autoComplete="username"
                  value={suUsername}
                  onChange={(e) => setSuUsername(e.target.value)}
                  required
                />
              </label>

              <label className="login__field">
                <span className="login__field-label">Email</span>
                <input
                  className="login__input"
                  type="email"
                  autoComplete="email"
                  value={suEmail}
                  onChange={(e) => setSuEmail(e.target.value)}
                  required
                />
              </label>

              <label className="login__field">
                <span className="login__field-label">Password</span>
                <input
                  className="login__input"
                  type="password"
                  autoComplete="new-password"
                  value={suPassword}
                  onChange={(e) => setSuPassword(e.target.value)}
                  required
                />
              </label>

              <label className="login__field">
                <span className="login__field-label">Confirm password</span>
                <input
                  className="login__input"
                  type="password"
                  autoComplete="new-password"
                  value={suConfirm}
                  onChange={(e) => setSuConfirm(e.target.value)}
                  required
                />
              </label>

              <button className="login__submit" type="submit" disabled={submitting}>
                {submitting ? 'Creating account…' : 'Create account'}
              </button>

              <div className="login__error" role="alert">
                {error}
              </div>
            </form>
          )}

          <FingerprintFooter deviceId={deviceId} sessionLabel={sessionLabel} />
        </div>
      </div>
    </div>
  )
}
