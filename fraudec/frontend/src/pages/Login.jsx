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

  const personas = [
    { name: 'Student', user_id: 'user_01' },
    { name: 'Doctor', user_id: 'user_02' },
    { name: 'Homeowner', user_id: 'user_03' },
    { name: 'Teacher', user_id: 'user_04' },
    { name: 'Software Engineer', user_id: 'user_05' },
    { name: 'Business Owner', user_id: 'user_06' },
    { name: 'Freelancer', user_id: 'user_07' },
    { name: 'Retired Person', user_id: 'user_08' },
    { name: 'Shop Owner', user_id: 'user_09' },
    { name: 'Corporate Employee', user_id: 'user_10' }
  ]

  async function handleDemoLogin(persona) {
    setError('')
    setSubmitting(true)

    try {
      const response = await fetch(`${API_BASE}/api/auth/demo-login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          persona_name: persona.name,
          user_id: persona.user_id,
        }),
      })

      if (!response.ok) {
        setError('Demo login failed. Is the backend running?')
        setSubmitting(false)
        return
      }
      const data = await response.json()
      completeAuth(data)
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
              aria-selected={mode === 'signin'}
              className={`login__tab ${mode === 'signin' ? 'login__tab--active' : ''}`}
              onClick={() => switchMode('signin')}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signup'}
              className={`login__tab ${mode === 'signup' ? 'login__tab--active' : ''}`}
              onClick={() => switchMode('signup')}
            >
              Create account
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'demo'}
              className={`login__tab ${mode === 'demo' ? 'login__tab--active' : ''}`}
              onClick={() => switchMode('demo')}
            >
              Demo Login
            </button>
          </div>

          <h1 className="login__title">{mode === 'demo' ? 'Select Persona' : (mode === 'signin' ? 'Sign in' : 'Create account')}</h1>
          <p className="login__subtitle">
            {mode === 'demo' ? 'Login instantly as a personalized synthetic user.' : (mode === 'signin' ? 'Analyst and admin access only' : 'Register for analyst access')}
          </p>

          {mode === 'demo' ? (
            <div className="login__demo-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '20px' }}>
              {personas.map(p => (
                <button 
                  key={p.user_id} 
                  className="login__submit" 
                  style={{ margin: 0, padding: '10px', fontSize: '14px', backgroundColor: '#1a1f36', border: '1px solid #3b4261', color: '#8f9bba' }}
                  onClick={() => handleDemoLogin(p)}
                  disabled={submitting}
                  onMouseOver={e => { e.currentTarget.style.backgroundColor = '#2a3150'; e.currentTarget.style.color = '#fff' }}
                  onMouseOut={e => { e.currentTarget.style.backgroundColor = '#1a1f36'; e.currentTarget.style.color = '#8f9bba' }}
                >
                  {p.name}
                </button>
              ))}
              <div className="login__error" role="alert" style={{ gridColumn: '1 / -1', textAlign: 'center' }}>
                {error}
              </div>
            </div>
          ) : mode === 'signin' ? (
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
