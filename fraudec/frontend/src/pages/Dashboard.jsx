import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import './Dashboard.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function Dashboard() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [activeTab, setActiveTab] = useState('live')
  const [transactions, setTransactions] = useState([])
  const [isSimulating, setIsSimulating] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [flaggedModal, setFlaggedModal] = useState(null)

  // Simulation state
  const simInterval = useRef(null)
  const simOffset = useRef(0)

  useEffect(() => {
    const token = localStorage.getItem('fraudec_token')
    if (!token) {
      navigate('/', { replace: true })
      return
    }

    try {
      const u = JSON.parse(localStorage.getItem('fraudec_user'))
      setUser(u)
      fetchMetrics(token)
    } catch {
      setUser(null)
    }
  }, [navigate])

  async function fetchMetrics(token) {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (res.ok) {
        setMetrics(await res.json())
      }
    } catch (e) {
      console.error(e)
    }
  }

  function handleSignOut() {
    localStorage.removeItem('fraudec_token')
    localStorage.removeItem('fraudec_user')
    navigate('/')
  }

  async function runSimulationStep() {
    const token = localStorage.getItem('fraudec_token')
    try {
      const res = await fetch(`${API_BASE}/api/transactions?limit=1&offset=${simOffset.current}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!res.ok) return
      const data = await res.json()
      if (data.transactions && data.transactions.length > 0) {
        const tx = data.transactions[0]
        simOffset.current += 1
        
        // Predict
        const predRes = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(tx)
        })
        if (predRes.ok) {
          const predData = await predRes.json()
          setTransactions(prev => [{ tx, pred: predData, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 50))
        }
      } else {
        setIsSimulating(false)
        clearInterval(simInterval.current)
      }
    } catch (e) {
      console.error(e)
    }
  }

  function toggleSimulation() {
    if (isSimulating) {
      clearInterval(simInterval.current)
      setIsSimulating(false)
    } else {
      setIsSimulating(true)
      simInterval.current = setInterval(runSimulationStep, 2500)
    }
  }

  function resetSimulation() {
    clearInterval(simInterval.current)
    setIsSimulating(false)
    simOffset.current = 0
    setTransactions([])
  }

  if (!user) return null

  const isDemo = user.role === 'persona'

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="brand-section">
          <span className="brand-logo">🛡️ BehaviorGuard</span>
          <span className="brand-subtitle">REAL-TIME FRAUD DETECTION</span>
        </div>
        <div className="user-section">
          <div className="persona-badge">
            <div className="persona-title">{isDemo ? user.username.toUpperCase() : 'ANALYST'}</div>
            <div className="persona-subtitle">{isDemo ? `${user.user_id} | Personalized Fraud Protection` : 'Admin Access'}</div>
          </div>
          <button className="signout-btn" onClick={handleSignOut}>Sign out</button>
        </div>
      </header>

      <nav className="dashboard-nav">
        <button className={activeTab === 'live' ? 'active' : ''} onClick={() => setActiveTab('live')}>Live Monitor</button>
        <button className={activeTab === 'analytics' ? 'active' : ''} onClick={() => setActiveTab('analytics')}>Analytics</button>
        <button className={activeTab === 'predict' ? 'active' : ''} onClick={() => setActiveTab('predict')}>Manual Predict</button>
        <button className={activeTab === 'evaluate' ? 'active' : ''} onClick={() => setActiveTab('evaluate')}>Model Evaluation</button>
      </nav>

      <main className="dashboard-content">
        {activeTab === 'live' && (
          <div className="live-monitor">
            <div className="stats-row">
              <div className="stat-card">
                <h3>Total Processed</h3>
                <div className="stat-val">{transactions.length}</div>
              </div>
              <div className="stat-card">
                <h3>Fraud Detected</h3>
                <div className="stat-val red">{transactions.filter(t => t.pred.prediction === 1).length}</div>
              </div>
              <div className="stat-card">
                <h3>Approval Rate</h3>
                <div className="stat-val green">
                  {transactions.length > 0 ? ((transactions.filter(t => t.pred.prediction === 0).length / transactions.length) * 100).toFixed(1) : '100'}%
                </div>
              </div>
            </div>

            <div className="sim-controls">
              <button className="btn-start" onClick={toggleSimulation}>{isSimulating ? '⏸ PAUSE' : '▶ START SIMULATION'}</button>
              <button className="btn-reset" onClick={resetSimulation}>🔄 RESET</button>
            </div>

            <div className="transaction-feed">
              {transactions.map((t, idx) => {
                const isFraud = t.pred.prediction === 1
                return (
                  <div key={idx} className={`tx-card ${isFraud ? 'tx-fraud' : 'tx-normal'}`} onClick={() => isFraud && setFlaggedModal(t)}>
                    <div className="tx-header">
                      <span className="tx-status">{isFraud ? '🚨 FRAUD DETECTED' : '✅ LIVE TRANSACTION'}</span>
                      <span className="tx-time">{t.time}</span>
                    </div>
                    <div className="tx-details">
                      <div><strong>TX ID:</strong> {t.tx.transaction_id || `TX-${simOffset.current - idx}`}</div>
                      <div><strong>Amount:</strong> ₹{t.tx.transaction_amount.toFixed(2)}</div>
                      <div><strong>Category:</strong> {t.tx.merchant_category}</div>
                      <div><strong>Location:</strong> {t.tx.city}</div>
                      <div><strong>Device:</strong> {t.tx.device_type}</div>
                    </div>
                    <div className="tx-footer">
                      <div className="tx-risk">Risk: {(t.pred.fraud_probability * 100).toFixed(1)}%</div>
                      <div className={`tx-decision ${isFraud ? 'blocked' : 'approved'}`}>
                        Status: {isFraud ? 'BLOCKED' : 'APPROVED'}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {activeTab === 'evaluate' && metrics && (
          <div className="model-evaluation">
            <h2>Model Performance Metrics for {user.username}</h2>
            <p className="eval-subtitle">Strict test set evaluation. 0% overlap with training data.</p>
            
            <div className="metrics-grid">
              <div className="metric-box">
                <div className="metric-name">F1 Score</div>
                <div className="metric-val">{(metrics.test_f1 * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-box">
                <div className="metric-name">Precision</div>
                <div className="metric-val">{(metrics.test_precision * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-box">
                <div className="metric-name">Recall</div>
                <div className="metric-val">{(metrics.test_recall * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-box">
                <div className="metric-name">ROC-AUC</div>
                <div className="metric-val">{(metrics.test_roc_auc * 100).toFixed(2)}%</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="analytics-view">
            <h2>Session Analytics</h2>
            <p>Analytics based on current simulation session.</p>
            <div className="stats-row">
               <div className="stat-card">
                  <h3>Avg Transaction</h3>
                  <div className="stat-val">₹{transactions.length > 0 ? (transactions.reduce((acc, curr) => acc + curr.tx.transaction_amount, 0) / transactions.length).toFixed(2) : '0.00'}</div>
               </div>
               <div className="stat-card">
                  <h3>Highest Risk Score</h3>
                  <div className="stat-val">{transactions.length > 0 ? (Math.max(...transactions.map(t => t.pred.fraud_probability)) * 100).toFixed(1) : '0.0'}%</div>
               </div>
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <ManualPredict />
        )}
      </main>

      {flaggedModal && (
        <div className="modal-overlay" onClick={() => setFlaggedModal(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header red-bg">
              <h2>🚨 FRAUD DETECTED</h2>
              <button onClick={() => setFlaggedModal(null)}>✖</button>
            </div>
            <div className="modal-body">
              <div className="modal-top">
                <div><strong>Probability:</strong> {(flaggedModal.pred.fraud_probability * 100).toFixed(1)}%</div>
                <div><strong>Decision:</strong> BLOCKED</div>
                <div><strong>Risk Level:</strong> CRITICAL</div>
              </div>
              <h3>Key Model Features</h3>
              <p className="feature-desc">The Random Forest model identified these features as the strongest indicators for this persona.</p>
              
              <div className="feature-bars">
                {flaggedModal.pred.top_features.map((f, i) => (
                  <div key={i} className="f-bar-row">
                    <div className="f-name">{f.feature}</div>
                    <div className="f-bar-track">
                      <div className="f-bar-fill" style={{ width: `${Math.min(100, f.importance * 300)}%` }}></div>
                    </div>
                    <div className="f-val">{(f.importance).toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ManualPredict() {
  const [formData, setFormData] = useState({
    transaction_amount: 500, merchant_category: 'Electronics', city: 'Bangalore', device_type: 'Mobile'
  })
  const [result, setResult] = useState(null)

  async function handlePredict() {
    const token = localStorage.getItem('fraudec_token')
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      if (res.ok) setResult(await res.json())
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="manual-predict">
      <h2>Manual Transaction Check</h2>
      <p>Enter details to query the model instantly.</p>
      
      <div className="predict-form">
        <label>
          Amount
          <input type="number" value={formData.transaction_amount} onChange={e => setFormData({...formData, transaction_amount: parseFloat(e.target.value)})} />
        </label>
        <label>
          Category
          <input type="text" value={formData.merchant_category} onChange={e => setFormData({...formData, merchant_category: e.target.value})} />
        </label>
        <label>
          City
          <input type="text" value={formData.city} onChange={e => setFormData({...formData, city: e.target.value})} />
        </label>
        <button onClick={handlePredict}>CHECK TRANSACTION</button>
      </div>

      {result && (
        <div className={`predict-result ${result.prediction === 1 ? 'pred-fraud' : 'pred-normal'}`}>
          <h3>{result.prediction === 1 ? '🔴 FRAUD DETECTED' : '✅ NORMAL TRANSACTION'}</h3>
          <div>Fraud Probability: {(result.fraud_probability * 100).toFixed(1)}%</div>
          <div>Decision: {result.prediction === 1 ? 'BLOCK' : 'APPROVE'}</div>
        </div>
      )}
    </div>
  )
}
