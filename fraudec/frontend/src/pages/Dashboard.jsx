import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import './Dashboard.css'

const API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const PROFILES = Array.from({ length: 10 }, (_, i) => `user_${String(i + 1).padStart(2, '0')}`)
const PROFILE_STORAGE_KEY = 'fraudec_profile'
const PAGE_SIZE = 15
const NUM = new Intl.NumberFormat('en-IN')
const INR = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 })
const FEATURE_GROUPS = [
  { title: 'Transaction', fields: ['transaction_amount', 'merchant_category', 'merchant_id', 'payment_method'] },
  { title: 'Device', fields: ['device_id', 'device_type'] },
  { title: 'Location / time', fields: ['city', 'hour_of_day', 'day_of_week', 'is_weekend'] },
  { title: 'Behavioral context', fields: ['transaction_gap_minutes', 'daily_transaction_count', 'average_amount_last_7_days', 'std_amount_last_7_days', 'merchant_visit_frequency', 'device_usage_frequency', 'location_visit_frequency'] },
  { title: 'Novelty / distance', fields: ['new_device', 'new_location', 'new_merchant', 'distance_from_last_transaction_km'] },
]
const NUMERIC_FIELDS = new Set(['transaction_amount', 'hour_of_day', 'day_of_week', 'transaction_gap_minutes', 'daily_transaction_count', 'average_amount_last_7_days', 'std_amount_last_7_days', 'merchant_visit_frequency', 'device_usage_frequency', 'location_visit_frequency', 'distance_from_last_transaction_km'])
const BOOLEAN_FIELDS = new Set(['is_weekend', 'new_device', 'new_location', 'new_merchant'])
const CHOICE_FIELDS = new Set(['merchant_category', 'payment_method', 'device_type', 'city'])
const LABELS = {
  transaction_amount: 'Amount', merchant_category: 'Merchant category', merchant_id: 'Merchant ID', payment_method: 'Payment method', device_id: 'Device ID', device_type: 'Device type', city: 'City', hour_of_day: 'Hour of day', day_of_week: 'Day of week (0 = Monday)', is_weekend: 'Weekend', transaction_gap_minutes: 'Transaction gap (minutes)', daily_transaction_count: 'Daily transaction count', average_amount_last_7_days: 'Average amount, last 7 days', std_amount_last_7_days: 'Amount standard deviation', merchant_visit_frequency: 'Merchant visit frequency', device_usage_frequency: 'Device usage frequency', location_visit_frequency: 'Location visit frequency', new_device: 'New device', new_location: 'New location', new_merchant: 'New merchant', distance_from_last_transaction_km: 'Distance from last transaction (km)',
}

function Brand() { return <span className="brand-mark"><span className="brand-mark__symbol">F</span><span className="brand-mark__word">FRAUDEC</span></span> }
function Notice({ title, children, retry }) { return <div className="dashboard-state-message" role={retry ? 'alert' : 'status'}><span className="dashboard-state-message__mark">{retry ? '!' : '⌁'}</span><div><strong>{title}</strong><p>{children}</p></div>{retry && <button className="dashboard-retry" onClick={retry}>Retry</button>}</div> }
function Loading() { return <div className="dashboard-loading" role="status"><span className="dashboard-spinner" /><span className="dashboard-loading__copy">Loading data from FraudEC API…</span></div> }
function Card({ label, value, detail, accent }) { return <article className={`dashboard-metric-card${accent ? ` dashboard-metric-card--${accent}` : ''}`}><span className="dashboard-metric-card__label">{label}</span><strong className="dashboard-metric-card__value">{value}</strong>{detail && <span className="dashboard-metric-card__detail">{detail}</span>}</article> }
function Heading({ eyebrow, title, description, aside }) { return <div className="dashboard-section__heading"><div><div className="dashboard-eyebrow">{eyebrow}</div><h2>{title}</h2><p>{description}</p></div>{aside}</div> }
function LineChart({ data, value, format = (n) => NUM.format(n), color = '#5ad8d0' }) {
  if (!data?.length) return <p className="dashboard-chart-empty">No time series returned for this profile.</p>
  const w = 700, h = 220, pad = 14, bottom = 30
  const vals = data.map((d) => Number(d[value]) || 0), max = Math.max(...vals, 1), plotH = h - bottom - pad
  const pts = vals.map((n, i) => `${pad + (data.length === 1 ? (w - pad * 2) / 2 : i * (w - pad * 2) / (data.length - 1))},${pad + plotH - n / max * plotH}`).join(' ')
  const mark = [...new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])]
  return <div className="dashboard-chart"><div className="dashboard-chart__range"><span>Peak {format(max)}</span><span>{data[0].date} — {data.at(-1).date}</span></div><svg className="dashboard-line-chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Historical transaction time series">{[0, 1, 2, 3].map((n) => <line key={n} x1={pad} x2={w - pad} y1={pad + plotH * n / 3} y2={pad + plotH * n / 3} className="dashboard-chart__gridline" />)}<polyline points={pts} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />{mark.map((i) => <circle key={i} cx={pad + (data.length === 1 ? (w - pad * 2) / 2 : i * (w - pad * 2) / (data.length - 1))} cy={pad + plotH - vals[i] / max * plotH} r="4" fill="#10191f" stroke={color} strokeWidth="2"><title>{data[i].date}: {format(vals[i])}</title></circle>)}</svg><div className="dashboard-chart__dates">{mark.map((i) => <span key={i}>{data[i].date}</span>)}</div></div>
}
function Bars({ data, color = 'green' }) {
  if (!data?.length) return <p className="dashboard-chart-empty">No breakdown returned.</p>
  const max = Math.max(...data.map((d) => Number(d.count) || 0), 1)
  return <div className="dashboard-bar-list">{data.map((d) => <div className="dashboard-bar-row" key={d.key}><span className="dashboard-bar-row__label" title={d.key}>{d.key}</span><span className="dashboard-bar-row__track"><span className={`dashboard-bar-row__fill dashboard-bar-row__fill--${color}`} style={{ width: `${d.count / max * 100}%` }} /></span><strong className="dashboard-bar-row__value">{NUM.format(d.count)}</strong></div>)}</div>
}
function Histogram({ data, title }) {
  if (!data?.length) return <p className="dashboard-chart-empty">No distribution returned.</p>
  const max = Math.max(...data.map((x) => Number(x.count) || 0), 1)
  return <div className="dashboard-histogram" role="img" aria-label={title}><div className="dashboard-histogram__bars">{data.map((x) => <div className="dashboard-histogram__column" key={x.range} title={`${x.range}: ${NUM.format(x.count)}`}><span className="dashboard-histogram__bar" style={{ height: `${Math.max(3, x.count / max * 100)}%` }} /></div>)}</div><div className="dashboard-histogram__labels"><span>{data[0].range}</span><span>{data.at(-1).range}</span></div></div>
}

export default function Dashboard() {
  const navigate = useNavigate(), location = useLocation()
  const active = location.pathname.split('/').filter(Boolean)[1] || 'live'
  const [user, setUser] = useState(null)
  const [profile, setProfile] = useState('')
  const [profileBusy, setProfileBusy] = useState(false)
  const [profileError, setProfileError] = useState('')
  const [retryKey, setRetryKey] = useState(0)
  const [overview, setOverview] = useState({ loading: true, data: null, error: '' })
  const [behavior, setBehavior] = useState({ loading: true, data: null, error: '' })
  const [rows, setRows] = useState({ loading: true, data: null, error: '' })
  const [monitor, setMonitor] = useState({ loading: false, data: null, error: '' })
  const [monitorPage, setMonitorPage] = useState(1)
  const [monitorLabel, setMonitorLabel] = useState('all')
  const [monitorSearch, setMonitorSearch] = useState('')
  const [monitorQuery, setMonitorQuery] = useState('')
  const [monitorRefresh, setMonitorRefresh] = useState(0)
  const [page, setPage] = useState(1), [label, setLabel] = useState('all'), [search, setSearch] = useState(''), [querySearch, setQuerySearch] = useState(''), [sort, setSort] = useState({ by: 'timestamp', order: 'desc' })
  const [options, setOptions] = useState(null)
  const [form, setForm] = useState(Object.fromEntries(FEATURE_GROUPS.flatMap((g) => g.fields.map((f) => [f, '']))))
  const [prediction, setPrediction] = useState({ loading: false, error: '', data: null })

  const request = useCallback(async (path, signal) => {
    const token = localStorage.getItem('fraudec_token')
    const response = await fetch(`${API}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {}, signal })
    const data = await response.json().catch(() => null)
    if (response.status === 401) {
      localStorage.removeItem('fraudec_token'); localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY)
      setProfile(''); setUser(null); navigate('/', { replace: true })
    }
    if (!response.ok) throw new Error(data?.detail || `Request failed (${response.status})`)
    return data
  }, [navigate])

  useEffect(() => {
    const token = localStorage.getItem('fraudec_token')
    if (!token) {
      localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY)
      navigate('/', { replace: true }); return
    }
    request('/api/auth/me').then(async (me) => {
      setUser(me)
      let stored = {}
      try { stored = JSON.parse(localStorage.getItem('fraudec_user') || '{}') } catch { /* Ignore stale local identity cache. */ }
      // The profile is a separate persisted value because /auth/me returns the account, not token claims.
      const savedProfile = localStorage.getItem(PROFILE_STORAGE_KEY)
      const selected = PROFILES.includes(savedProfile) ? savedProfile : stored.user_id || ''
      if (!PROFILES.includes(selected)) { localStorage.removeItem(PROFILE_STORAGE_KEY); setProfile(''); return }

      // Re-issue the selected-profile token on refresh so local profile state and
      // backend authorization are synchronized before profile data is requested.
      const response = await fetch(`${API}/api/auth/select-demo-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ user_id: selected }),
      })
      const result = await response.json().catch(() => null)
      if (response.status === 401) throw new Error('Your session has expired. Please sign in again.')
      if (!response.ok || !result?.access_token || result?.user?.user_id !== selected) {
        localStorage.removeItem(PROFILE_STORAGE_KEY); setProfile('')
        return
      }
      localStorage.setItem('fraudec_token', result.access_token)
      localStorage.setItem('fraudec_user', JSON.stringify(result.user))
      localStorage.setItem(PROFILE_STORAGE_KEY, selected)
      setUser((current) => ({ ...current, ...result.user }))
      setProfile(selected)
    }).catch(() => {
      localStorage.removeItem('fraudec_token'); localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY)
      setProfile(''); navigate('/', { replace: true })
    })
  }, [navigate, request])

  useEffect(() => {
    if (!user || !profile || active !== 'analytics') return undefined
    const controller = new AbortController()
    setOverview({ loading: true, data: null, error: '' }); setBehavior({ loading: true, data: null, error: '' }); setRows({ loading: true, data: null, error: '' })
    request('/api/dashboard/overview', controller.signal).then((data) => setOverview({ loading: false, data, error: '' })).catch((e) => { if (e.name !== 'AbortError') setOverview({ loading: false, data: null, error: e.message }) })
    request('/api/dashboard/behavior', controller.signal).then((data) => setBehavior({ loading: false, data, error: '' })).catch((e) => { if (e.name !== 'AbortError') setBehavior({ loading: false, data: null, error: e.message }) })
    return () => controller.abort()
  }, [user, profile, active, retryKey, request])

  useEffect(() => { const timer = setTimeout(() => setMonitorQuery(monitorSearch.trim()), 250); return () => clearTimeout(timer) }, [monitorSearch])
  useEffect(() => {
    if (!user || !profile || active !== 'live') return undefined
    const controller = new AbortController()
    const params = new URLSearchParams({ page: String(monitorPage), page_size: '15', sort_by: 'timestamp', sort_order: 'desc' })
    if (monitorLabel !== 'all') params.set('label', monitorLabel)
    if (monitorQuery) params.set('search', monitorQuery)
    setMonitor({ loading: true, data: null, error: '' })
    request(`/api/transactions?${params}`, controller.signal)
      .then((data) => setMonitor({ loading: false, data, error: '' }))
      .catch((error) => { if (error.name !== 'AbortError') setMonitor({ loading: false, data: null, error: error.message }) })
    return () => controller.abort()
  }, [user, profile, active, monitorPage, monitorLabel, monitorQuery, monitorRefresh, request])

  useEffect(() => { const timer = setTimeout(() => setQuerySearch(search.trim()), 250); return () => clearTimeout(timer) }, [search])
  useEffect(() => {
    if (!user || !profile || active !== 'analytics') return undefined
    const controller = new AbortController(), params = new URLSearchParams({ page: String(page), page_size: String(PAGE_SIZE), sort_by: sort.by, sort_order: sort.order })
    if (label !== 'all') params.set('label', label)
    if (querySearch) params.set('search', querySearch)
    setRows({ loading: true, data: null, error: '' })
    request(`/api/transactions?${params}`, controller.signal).then((data) => { setRows({ loading: false, data, error: '' }); if (page > Math.max(1, data.total_pages)) setPage(Math.max(1, data.total_pages)) }).catch((e) => { if (e.name !== 'AbortError') setRows({ loading: false, data: null, error: e.message }) })
    return () => controller.abort()
  }, [user, profile, active, page, label, querySearch, sort, retryKey, request])

  useEffect(() => {
    if (!user || !profile || active !== 'predict') return undefined
    const controller = new AbortController(); setOptions(null)
    request('/api/prediction/options', controller.signal).then(setOptions).catch((e) => { if (e.name !== 'AbortError') setPrediction((s) => ({ ...s, error: e.message })) })
    return () => controller.abort()
  }, [user, profile, active, retryKey, request])

  async function chooseProfile(next) {
    if (!PROFILES.includes(next) || profileBusy || next === profile) return
    setProfileBusy(true); setProfileError('')
    try {
      // Profile authorization is a deliberate, allowlisted choice scoped into a new token.
      const response = await fetch(`${API}/api/auth/select-demo-profile`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('fraudec_token')}` }, body: JSON.stringify({ user_id: next }) })
      const result = await response.json().catch(() => null)
      if (response.status === 401) {
        localStorage.removeItem('fraudec_token'); localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY)
        setProfile(''); setUser(null); navigate('/', { replace: true })
      }
      if (!response.ok) throw new Error(result?.detail || 'Could not select dataset profile.')
      if (!result?.access_token || result?.user?.user_id !== next) throw new Error('The server did not return a profile-scoped session.')
      localStorage.setItem('fraudec_token', result.access_token)
      localStorage.setItem('fraudec_user', JSON.stringify(result.user))
      localStorage.setItem(PROFILE_STORAGE_KEY, next)
      setProfile(next); setUser((current) => ({ ...current, ...result.user })); setPage(1); setMonitorPage(1); setOptions(null); setPrediction({ loading: false, data: null, error: '' })
    } catch (e) { setProfileError(e.message || 'Could not select dataset profile.') } finally { setProfileBusy(false) }
  }

  async function signOut() {
    const token = localStorage.getItem('fraudec_token')
    try { await fetch(`${API}/api/auth/logout`, { method: 'POST', headers: token ? { Authorization: `Bearer ${token}` } : {} }) } catch { /* Local sign-out still clears the browser session. */ }
    localStorage.removeItem('fraudec_token'); localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY); navigate('/', { replace: true })
  }

  async function submitPrediction(event) {
    event.preventDefault(); setPrediction({ loading: true, data: null, error: '' })
    const body = {}
    for (const group of FEATURE_GROUPS) for (const field of group.fields) {
      const raw = form[field]
      body[field] = BOOLEAN_FIELDS.has(field) || NUMERIC_FIELDS.has(field) ? Number(raw) : raw
    }
    try { const response = await fetch(`${API}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('fraudec_token')}` }, body: JSON.stringify(body) })
      const result = await response.json().catch(() => null)
      if (response.status === 401) {
        localStorage.removeItem('fraudec_token'); localStorage.removeItem('fraudec_user'); localStorage.removeItem(PROFILE_STORAGE_KEY)
        setProfile(''); setUser(null); navigate('/', { replace: true })
      }
      if (!response.ok) throw new Error(result?.detail || `Prediction failed (${response.status})`)
      setPrediction({ loading: false, data: result, error: '' })
    } catch (e) { setPrediction({ loading: false, data: null, error: e.message || 'Prediction failed.' }) }
  }

  const metrics = useMemo(() => overview.data && [
    ['Historical transactions', NUM.format(overview.data.total_transaction_count)],
    ['Fraud-labelled transactions', NUM.format(overview.data.fraud_labelled_count)],
    ['Normal transactions', NUM.format(overview.data.normal_count)],
    ['Historical fraud labels', `${overview.data.fraud_labelled_percentage.toFixed(2)}%`],
    ['Total transaction amount', INR.format(overview.data.total_transaction_amount)],
    ['Average transaction amount', INR.format(overview.data.average_transaction_amount)],
  ], [overview.data])
  const monitorSummary = useMemo(() => {
    const transactions = monitor.data?.transactions || []
    const latest = transactions.reduce((max, transaction) => {
      const timestamp = Date.parse(transaction.timestamp)
      return Number.isFinite(timestamp) && timestamp > max ? timestamp : max
    }, 0)
    return {
      loaded: transactions.length,
      fraud: transactions.filter((transaction) => Number(transaction.label) === 1).length,
      normal: transactions.filter((transaction) => Number(transaction.label) === 0).length,
      latest: latest ? new Date(latest).toLocaleString() : 'Unavailable',
    }
  }, [monitor.data])

  if (!user) return <div className="dashboard-loading"><span className="dashboard-spinner" />Checking session…</div>
  const email = user.email || 'Authenticated user'
  const profileName = profile ? profile.replace('user_', 'Profile ') : 'No dataset selected'

  return <div className="dashboard-shell">
    <aside className="dashboard-sidebar" aria-label="Primary navigation"><a className="dashboard-sidebar__brand" href="/dashboard/live"><Brand /></a><div className="dashboard-sidebar__section-label">FRAUD INTELLIGENCE</div><nav className="dashboard-sidebar__nav">
      {[['live', 'Live Monitor', '◉'], ['predict', 'Manual Prediction', '⌁'], ['analytics', 'Analytics', '▦']].map(([key, name, icon]) => <a key={key} href={`/dashboard/${key}`} onClick={(e) => { e.preventDefault(); navigate(`/dashboard/${key}`) }} className={`dashboard-nav-link${active === key ? ' dashboard-nav-link--active' : ''}`} aria-current={active === key ? 'page' : undefined}><span className="dashboard-nav-link__icon">{icon}</span>{name}</a>)}
    </nav><div className="dashboard-sidebar__bottom"><div className="dashboard-sidebar__status"><span /> Historical dataset workspace</div><span className="dashboard-sidebar__version">FRAUDEC · ML / FRAUD DETECTION</span></div></aside>
    <div className="dashboard-main-column"><header className="dashboard-topbar"><div className="dashboard-topbar__context"><span>Workspace</span><span className="dashboard-topbar__slash">/</span><strong>{active === 'predict' ? 'Manual Prediction' : active === 'analytics' ? 'Analytics' : 'Live Monitor'}</strong></div><div className="dashboard-topbar__account"><label className="dashboard-profile-picker"><span>DATASET PROFILE</span><select value={profile} onChange={(e) => chooseProfile(e.target.value)} disabled={profileBusy} aria-label="Select allowlisted demo dataset"><option value="">Select profile</option>{PROFILES.map((p) => <option key={p} value={p}>{p.replace('user_', 'Profile ')}</option>)}</select></label><div className="dashboard-topbar__identity"><span className="dashboard-avatar">{email.slice(0, 1).toUpperCase()}</span><span className="dashboard-topbar__user-copy"><strong>{email}</strong><span>{user.role || 'authenticated'}</span></span></div><button className="dashboard-signout" type="button" onClick={signOut}>Sign out</button></div></header>
      <main className="dashboard-content">
        <section className="dashboard-welcome"><div className="dashboard-welcome__copy"><div className="dashboard-eyebrow"><span /> {active === 'live' ? 'LIVE MONITOR' : 'FRAUDEC SECURITY WORKSPACE'}</div><h1>{active === 'live' ? 'Live Monitor' : active === 'predict' ? 'Manual Prediction' : 'Behavioral Analytics'}</h1><p>{active === 'live' ? profile ? `${profileName} / currently selected profile` : 'Select a dataset profile to monitor transaction activity.' : active === 'predict' ? 'Submit a complete transaction context to the profile’s Random Forest model.' : `Historical transaction behavior for ${profileName}.`}</p></div><div className="dashboard-welcome__profile"><span>DATA SOURCE</span><strong>{profile ? `${profileName} historical dataset` : 'Not selected'}</strong></div></section>
        {profileError && <Notice title="Profile selection failed" retry={() => setProfileError('')}>{profileError}</Notice>}

        {active === 'live' && <section className="dashboard-section dashboard-monitor">
          <div className="dashboard-monitor-mode"><div><span className="dashboard-chart-card__eyebrow">HISTORICAL DATA MODE</span><p>Displaying recent records from the selected profile’s historical transaction dataset. No live ingestion stream is currently connected.</p></div><button className="dashboard-monitor-refresh" type="button" onClick={() => setMonitorRefresh((value) => value + 1)} disabled={!profile || monitor.loading}>{monitor.loading ? 'Loading dataset…' : 'Refresh dataset'}</button></div>
          {!profile ? <Notice title="Select a dataset profile to monitor transaction activity.">Choose one of the supported profiles from the selector above.</Notice> : <>
            <div className="dashboard-metrics-grid dashboard-monitor-summary"><Card label="Transactions loaded" value={monitor.loading && !monitor.data ? '—' : NUM.format(monitorSummary.loaded)} detail="Records on this page"/><Card label="Fraud-labelled records" value={monitor.loading && !monitor.data ? '—' : NUM.format(monitorSummary.fraud)} detail="Historical dataset label" accent="amber"/><Card label="Legitimate-labelled records" value={monitor.loading && !monitor.data ? '—' : NUM.format(monitorSummary.normal)} detail="Historical dataset label"/><Card label="Latest dataset timestamp" value={monitor.loading && !monitor.data ? '—' : monitorSummary.latest} detail="From records on this page"/></div>
            <article className="dashboard-table-card dashboard-monitor-feed"><div className="dashboard-monitor-feed__header"><div><span className="dashboard-chart-card__eyebrow">{profileName.toUpperCase()} · HISTORICAL RECORDS</span><h2>Recent Transaction Activity</h2><p>Dataset labels show the existing ground-truth label, not a model prediction.</p></div><span className="dashboard-history-badge"><i/> Historical Dataset Label</span></div>
              <div className="dashboard-table-toolbar"><label className="dashboard-search"><span>⌕</span><input type="search" value={monitorSearch} onChange={(event) => { setMonitorSearch(event.target.value); setMonitorPage(1) }} placeholder="Search transaction, category, method, device or city" aria-label="Search historical transactions"/></label><label className="dashboard-filter"><span>Dataset Label</span><select value={monitorLabel} onChange={(event) => { setMonitorLabel(event.target.value); setMonitorPage(1) }}><option value="all">All labels</option><option value="0">Normal</option><option value="1">Fraud-labelled</option></select></label></div>
              {monitor.loading ? <Loading/> : monitor.error ? <Notice title="Historical transactions unavailable" retry={() => setMonitorRefresh((value) => value + 1)}>{monitor.error}</Notice> : monitor.data?.transactions?.length ? <><div className="dashboard-monitor-list">{monitor.data.transactions.map((transaction) => <article className={`dashboard-monitor-row${Number(transaction.label) === 1 ? ' dashboard-monitor-row--fraud' : ''}`} key={transaction.transaction_id}><div className="dashboard-monitor-row__main"><div className="dashboard-monitor-row__identity"><strong>{transaction.transaction_id}</strong><time dateTime={transaction.timestamp}>{new Date(transaction.timestamp).toLocaleString()}</time></div><div className="dashboard-monitor-row__details"><span><small>Amount</small><strong>{INR.format(transaction.transaction_amount)}</strong></span><span><small>Merchant category</small><strong>{transaction.merchant_category}</strong></span><span><small>Payment method</small><strong>{transaction.payment_method}</strong></span><span><small>Device / city</small><strong>{transaction.device_type} · {transaction.city}</strong></span></div></div><span className={`dashboard-label-badge${Number(transaction.label) === 1 ? ' dashboard-label-badge--fraud' : ''}`}>{Number(transaction.label) === 1 ? 'Fraud-labelled' : 'Normal'}</span></article>)}</div><div className="dashboard-table-pagination"><span>Page {monitor.data.page} of {monitor.data.total_pages} · {NUM.format(monitor.data.total)} matching records</span><div><button type="button" disabled={monitorPage <= 1 || monitor.loading} onClick={() => setMonitorPage((pageNumber) => Math.max(1, pageNumber - 1))}>Previous</button><span>{monitorPage} / {monitor.data.total_pages}</span><button type="button" disabled={monitorPage >= monitor.data.total_pages || monitor.loading} onClick={() => setMonitorPage((pageNumber) => pageNumber + 1)}>Next</button></div></div></> : <div className="dashboard-table-empty"><strong>No historical transactions found</strong><p>There are no records matching the current search and dataset-label filter.</p></div>}
            </article>
          </>}
        </section>}

        {active === 'predict' && <section className="dashboard-section"><Heading eyebrow="MODEL INFERENCE" title="Manual Prediction" description="Every required model feature is supplied explicitly. Behavioral fields are the context you provide to the model; no defaults are inferred." aside={<span className="dashboard-feature-note">Random Forest · selected profile</span>} />{!profile ? <Notice title="Select a dataset profile">An allowlisted profile is required to choose its matching preprocessing and Random Forest artifacts.</Notice> : prediction.error && !options ? <Notice title="Prediction inputs could not be loaded" retry={() => setRetryKey((n) => n + 1)}>{prediction.error}</Notice> : !options ? <Loading /> : <form className="dashboard-prediction-form" onSubmit={submitPrediction}>{FEATURE_GROUPS.map((group) => <fieldset className="dashboard-form-group" key={group.title}><legend>{group.title}</legend><div className="dashboard-form-grid">{group.fields.map((field) => <label className="dashboard-form-field" key={field}><span>{LABELS[field]}</span>{BOOLEAN_FIELDS.has(field) ? <select value={form[field]} required onChange={(e) => setForm((f) => ({ ...f, [field]: e.target.value }))}><option value="">Choose value</option><option value="0">No / False</option><option value="1">Yes / True</option></select> : CHOICE_FIELDS.has(field) ? <select value={form[field]} required onChange={(e) => setForm((f) => ({ ...f, [field]: e.target.value }))}><option value="">Choose actual dataset value</option>{(options[field] || []).map((v) => <option key={v} value={v}>{v}</option>)}</select> : <input type={NUMERIC_FIELDS.has(field) ? 'number' : 'text'} step={NUMERIC_FIELDS.has(field) && !['hour_of_day', 'day_of_week', 'daily_transaction_count'].includes(field) ? 'any' : undefined} min={field === 'hour_of_day' ? 0 : field === 'day_of_week' ? 0 : undefined} max={field === 'hour_of_day' ? 23 : field === 'day_of_week' ? 6 : undefined} list={field === 'merchant_id' || field === 'device_id' ? `${field}-values` : undefined} value={form[field]} required onChange={(e) => setForm((f) => ({ ...f, [field]: e.target.value }))} placeholder={field === 'merchant_id' || field === 'device_id' ? 'Enter actual ID' : ''} />}{(field === 'merchant_id' || field === 'device_id') && <datalist id={`${field}-values`}>{(options[field] || []).map((v) => <option key={v} value={v} />)}</datalist>}</label>)}</div></fieldset>)}<div className="dashboard-prediction-submit"><p>Prediction uses the selected profile’s trained Random Forest and matching preprocessor.</p><button className="dashboard-primary-button" disabled={prediction.loading}>{prediction.loading ? 'Running model…' : 'Run prediction'}</button></div>{prediction.error && <Notice title="Prediction request failed" retry={() => setPrediction((p) => ({ ...p, error: '' }))}>{prediction.error}</Notice>}</form>}{prediction.data && <article className={`dashboard-prediction-result${prediction.data.prediction === 1 ? ' dashboard-prediction-result--fraud' : ''}`} aria-live="polite"><span className="dashboard-chart-card__eyebrow">MODEL RESPONSE</span><h3>{prediction.data.prediction === 1 ? 'Fraud' : 'Legitimate'}</h3><p>Fraud probability <strong>{(prediction.data.fraud_probability * 100).toFixed(2)}%</strong></p><h4>Top Model Features</h4><div className="dashboard-feature-list">{prediction.data.top_features?.map((item) => <div key={item.feature}><span>{item.feature}</span><strong>{Number(item.importance).toFixed(5)}</strong></div>)}</div><small>Feature importance is model-level and is not a transaction-specific SHAP explanation.</small></article>}</section>}

        {active === 'analytics' && (!profile ? <section className="dashboard-section"><Notice title="Choose a dataset profile">Analytics are available for the ten allowlisted demo datasets. Choose a profile to load real historical records.</Notice></section> : <>
          <section className="dashboard-section" id="overview"><Heading eyebrow="01 · HISTORICAL DATA" title="Transaction Overview" description="Historical transactions and synthetic ground-truth dataset labels; these are not model detections." aside={<span className="dashboard-label-note"><span/> Historical Fraud Labels</span>} />{overview.loading ? <Loading /> : overview.error ? <Notice title="Overview unavailable" retry={() => setRetryKey((n) => n + 1)}>{overview.error}</Notice> : <><div className="dashboard-metrics-grid">{metrics?.map(([labelText, valueText], i) => <Card key={labelText} label={labelText} value={valueText} detail={i === 1 || i === 2 || i === 3 ? 'Synthetic dataset labels' : 'From selected historical dataset'} accent={i === 1 || i === 3 ? 'amber' : ''}/>)}</div><div className="dashboard-chart-grid dashboard-chart-grid--overview"><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">HISTORICAL ACTIVITY</span><h3>Transaction volume over time</h3></div></div><LineChart data={overview.data.transaction_volume_over_time} value="transaction_count" /></article><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">HISTORICAL AMOUNT</span><h3>Transaction amount over time</h3></div></div><LineChart data={overview.data.transaction_amount_over_time} value="transaction_amount" color="#52b9db" format={INR.format} /></article></div></>}</section>
          <section className="dashboard-section" id="behavior"><Heading eyebrow="02 · BEHAVIORAL FEATURES" title="Behavioral Analytics" description="Descriptive transaction and behavioral features from the dataset, not fraud predictions." />{behavior.loading ? <Loading /> : behavior.error ? <Notice title="Behavior unavailable" retry={() => setRetryKey((n) => n + 1)}>{behavior.error}</Notice> : <><div className="dashboard-chart-grid dashboard-chart-grid--behavior-primary"><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">24 HOUR DISTRIBUTION</span><h3>Transactions by hour</h3></div></div><Bars data={behavior.data.transactions_by_hour}/></article><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">WEEKLY RHYTHM</span><h3>Transactions by weekday</h3></div></div><Bars data={behavior.data.transactions_by_day_of_week} color="blue"/></article></div><div className="dashboard-chart-grid dashboard-chart-grid--mix">{[['Merchant categories', behavior.data.merchant_category_distribution, 'green'], ['Payment methods', behavior.data.payment_method_distribution, 'blue'], ['Device types', behavior.data.device_type_distribution, 'violet']].map(([title, data, color]) => <article className="dashboard-chart-card" key={title}><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">TRANSACTION MIX</span><h3>{title}</h3></div></div><Bars data={data} color={color}/></article>)}</div><article className="dashboard-feature-card"><div className="dashboard-feature-card__intro"><span className="dashboard-chart-card__eyebrow">NOVELTY FEATURES</span><h3>New activity flags</h3><p>Counts from the existing new-device, new-location and new-merchant fields.</p></div><div className="dashboard-feature-card__metrics"><Card label="New device activity" value={NUM.format(behavior.data.new_device_count)}/><Card label="New location activity" value={NUM.format(behavior.data.new_location_count)}/><Card label="New merchant activity" value={NUM.format(behavior.data.new_merchant_count)}/></div></article><div className="dashboard-chart-grid dashboard-chart-grid--distribution"><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">TIME BETWEEN EVENTS</span><h3>Transaction gap distribution</h3></div></div><Histogram data={behavior.data.transaction_gap_distribution} title="Transaction gap distribution"/></article><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">LOCATION MOVEMENT</span><h3>Distance from previous transaction</h3></div></div><Histogram data={behavior.data.distance_from_previous_transaction} title="Distance from previous transaction"/></article></div><div className="dashboard-chart-grid dashboard-chart-grid--trends"><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">DAILY ACTIVITY</span><h3>Daily transaction count</h3></div></div><LineChart data={behavior.data.daily_transaction_count_over_time} value="count" color="#65cadb"/></article><article className="dashboard-chart-card"><div className="dashboard-chart-card__heading"><div><span className="dashboard-chart-card__eyebrow">ROLLING FEATURE</span><h3>Average amount over last 7 days</h3></div></div><LineChart data={behavior.data.average_amount_last_7_days_over_time} value="average_amount" color="#c18be0" format={INR.format}/></article></div></>}</section>
          <section className="dashboard-section dashboard-section--review" id="review"><Heading eyebrow="03 · HISTORICAL RECORDS" title="Transaction Review" description="Paginated transaction rows with the dataset’s existing ground-truth label." aside={<span className="dashboard-label-note"><span/> Dataset Label</span>}/><div className="dashboard-table-card"><div className="dashboard-table-toolbar"><label className="dashboard-search"><span>⌕</span><input type="search" value={search} onChange={(e) => { setSearch(e.target.value); setPage(1) }} placeholder="Search transaction, category, method, device or city" aria-label="Search transactions"/></label><label className="dashboard-filter"><span>Dataset label</span><select value={label} onChange={(e) => { setLabel(e.target.value); setPage(1) }}><option value="all">All</option><option value="0">Normal</option><option value="1">Fraud-labelled</option></select></label></div>{rows.loading ? <Loading/> : rows.error ? <Notice title="Transactions unavailable" retry={() => setRetryKey((n) => n + 1)}>{rows.error}</Notice> : rows.data?.transactions.length ? <><div className="dashboard-table-scroll"><table className="dashboard-table"><thead><tr>{[['timestamp', 'Transaction / timestamp'], ['transaction_amount', 'Amount'], ['merchant_category', 'Merchant category'], ['', 'Payment method'], ['', 'Device'], ['', 'City'], ['label', 'Dataset Label']].map(([field, title], i) => <th key={`${title}${i}`}>{field ? <button type="button" onClick={() => { setSort((s) => ({ by: field, order: s.by === field && s.order === 'asc' ? 'desc' : 'asc' })); setPage(1) }}>{title} {sort.by === field ? (sort.order === 'asc' ? '↑' : '↓') : '↕'}</button> : title}</th>)}</tr></thead><tbody>{rows.data.transactions.map((tx) => <tr key={tx.transaction_id}><td><strong className="dashboard-table__id">{tx.transaction_id}</strong><span className="dashboard-table__timestamp">{new Date(tx.timestamp).toLocaleString()}</span></td><td className="dashboard-table__amount">{INR.format(tx.transaction_amount)}</td><td>{tx.merchant_category}</td><td>{tx.payment_method}</td><td>{tx.device_type}</td><td>{tx.city}</td><td><span className={`dashboard-label-badge${tx.label === 1 ? ' dashboard-label-badge--fraud' : ''}`}>{tx.label === 1 ? 'Fraud-labelled' : 'Normal'}</span></td></tr>)}</tbody></table></div><div className="dashboard-table-pagination"><span>Showing {NUM.format((page - 1) * PAGE_SIZE + 1)}–{NUM.format(Math.min(page * PAGE_SIZE, rows.data.total))} of {NUM.format(rows.data.total)}</span><div><button disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>Previous</button><span>Page {page} of {rows.data.total_pages}</span><button disabled={page >= rows.data.total_pages} onClick={() => setPage((p) => p + 1)}>Next</button></div></div></> : <div className="dashboard-table-empty"><strong>No records match these filters</strong><p>Change the search or label filter.</p></div>}</div></section>
        </>)}
        <footer className="dashboard-footer"><Brand/><span>Historical behavior, clearly labeled.</span></footer>
      </main></div></div>
}
