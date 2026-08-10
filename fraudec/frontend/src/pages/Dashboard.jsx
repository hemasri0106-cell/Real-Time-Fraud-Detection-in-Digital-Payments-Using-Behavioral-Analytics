import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './Dashboard.css'

export default function Dashboard() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)

  useEffect(() => {
    const token = localStorage.getItem('fraudec_token')
    if (!token) {
      navigate('/', { replace: true })
      return
    }

    try {
      setUser(JSON.parse(localStorage.getItem('fraudec_user')))
    } catch {
      setUser(null)
    }
  }, [navigate])

  function handleSignOut() {
    localStorage.removeItem('fraudec_token')
    localStorage.removeItem('fraudec_user')
    navigate('/')
  }

  if (!user) return null

  return (
    <div className="dashboard">
      <div className="dashboard__card">
        <span className="dashboard__brand">FRAUDEC</span>
        <p className="dashboard__welcome">
          Signed in as <strong>{user.username}</strong>
        </p>
        <p className="dashboard__role">Role: {user.role}</p>
        <button className="dashboard__signout" onClick={handleSignOut}>
          Sign out
        </button>
      </div>
    </div>
  )
}
