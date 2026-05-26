import React, { useState, useEffect } from 'react'

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [isRegistering, setIsRegistering] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [topic, setTopic] = useState('AI Agents in 2024')
  const [loading, setLoading] = useState(false)
  const [memory, setMemory] = useState({ research: {}, business_topic: {}, resources: [] })
  const [config, setConfig] = useState({ agents: [], tasks: [], tools: [] })
  const [logs, setLogs] = useState([])

  const fetchMemory = async () => {
    try {
      const response = await fetch('http://localhost:8000/memory')
      const data = await response.json()
      setMemory(data)
    } catch (err) {
      console.error("Failed to fetch memory", err)
    }
  }

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/config')
      const data = await response.json()
      setConfig(data)
    } catch (err) {
      console.error("Failed to fetch config", err)
    }
  }

  useEffect(() => {
    if (isLoggedIn) {
      fetchMemory()
      fetchConfig()
      const interval = setInterval(fetchMemory, 5000)
      return () => clearInterval(interval)
    }
  }, [isLoggedIn])

  const handleLogin = (e) => {
    e.preventDefault()
    if (username && password) {
      if (isRegistering) {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] Account created for ${username}!`, ...prev])
        setIsRegistering(false)
        alert("Registration successful! Please login.")
      } else {
        setIsLoggedIn(true)
      }
    }
  }

  const runAgent = async () => {
    setLoading(true)
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] Initializing Virtual Crew...`, ...prev])
    try {
      const response = await fetch(`http://localhost:8000/execute_task?topic=${encodeURIComponent(topic)}`, {
        method: 'POST'
      })
      const data = await response.json()
      const resultMsg = data.result ? data.result.substring(0, 100) + "..." : "Task completed successfully"
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] Success: ${resultMsg}`, ...prev])
    } catch (err) {
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] Error: ${err.message}`, ...prev])
    } finally {
      setLoading(false)
    }
  }

  if (!isLoggedIn) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h1>{isRegistering ? 'Create Account' : 'Welcome Back'}</h1>
          <p style={{color: 'var(--text-secondary)', marginBottom: '2rem'}}>
            {isRegistering ? 'Join the virtual workforce today' : 'Access your virtual employee dashboard'}
          </p>
          <form onSubmit={handleLogin}>
            <input 
              type="text" 
              placeholder="Username" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <input 
              type="password" 
              placeholder="Password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button type="submit">{isRegistering ? 'Register' : 'Login to Dashboard'}</button>
          </form>
          <p style={{marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', textAlign: 'center'}}>
            {isRegistering ? 'Already have an account?' : "Don't have an account?"} {' '}
            <span 
              onClick={() => setIsRegistering(!isRegistering)} 
              style={{color: 'var(--accent-primary)', cursor: 'pointer', fontWeight: 'bold'}}
            >
              {isRegistering ? 'Login' : 'Register'}
            </span>
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">
          <h1 style={{fontSize: '1.5rem'}}>Virtual Employee <span style={{opacity: 0.5}}>v9.0</span></h1>
        </div>
        <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
          <div className="status-badge">System Online</div>
          <button onClick={() => setIsLoggedIn(false)} style={{width: 'auto', padding: '0.4rem 1rem', background: 'transparent', border: '1px solid var(--glass-border)'}}>Logout</button>
        </div>
      </header>

      <main className="dashboard">
        <aside className="sidebar">
          <div className="card">
            <h2>Command Center</h2>
            <input 
              type="text" 
              value={topic} 
              onChange={(e) => setTopic(e.target.value)} 
              placeholder="Enter research topic..."
            />
            <button onClick={runAgent} disabled={loading}>
              {loading ? 'Processing...' : 'Run Virtual Crew'}
            </button>
          </div>

          <div className="card">
            <h2>Activity Logs</h2>
            <div style={{height: '300px', overflowY: 'auto', fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
              {logs.map((log, i) => (
                <div key={i} style={{marginBottom: '0.5rem', borderLeft: '2px solid var(--accent-primary)', paddingLeft: '0.5rem'}}>
                  {log}
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="main-content">
          <div className="card">
            <h2>System Architecture (Agents & Tasks)</h2>
            <div className="config-grid">
              {config.agents.map((agent, i) => (
                <div key={i} className="agent-pill">
                  <h3 style={{fontSize: '0.9rem', color: 'var(--accent-primary)'}}>{agent.role.toUpperCase()}</h3>
                  <p style={{fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem'}}>{agent.goal}</p>
                </div>
              ))}
            </div>
            <div className="config-grid" style={{marginTop: '1rem'}}>
              {config.tasks.map((task, i) => (
                <div key={i} className="task-pill">
                  <h3 style={{fontSize: '0.8rem', color: 'var(--accent-secondary)'}}>TASK {i+1}</h3>
                  <p style={{fontSize: '0.75rem', opacity: 0.7, marginTop: '0.3rem'}}>{task.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem'}}>
            <div className="card">
              <h2>Research Insights</h2>
              <div className="memory-section">
                {Object.entries(memory.business_topic).map(([key, val]) => (
                  <div key={key} style={{padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', marginBottom: '1rem'}}>
                    <h3 style={{fontSize: '1rem'}}>{key}</h3>
                    <p style={{fontSize: '0.85rem', opacity: 0.8, marginTop: '0.5rem'}}>{val.insights}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h2>Resource Management</h2>
              <div style={{display: 'flex', flexDirection: 'column', gap: '0.75rem'}}>
                {memory.resources.map((res, i) => (
                  <div key={i} style={{padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--glass-border)'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
                      <span style={{fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--accent-primary)'}}>{res.resource.toUpperCase()}</span>
                      <span className="status-badge" style={{
                        marginTop: 0,
                        background: res.priority === 'High' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                        color: res.priority === 'High' ? '#ef4444' : 'var(--success)'
                      }}>
                        {res.priority || 'Normal'}
                      </span>
                    </div>
                    <p style={{fontSize: '0.8rem', opacity: 0.8}}>{res.details}</p>
                    <div style={{fontSize: '0.7rem', opacity: 0.4, marginTop: '0.5rem'}}>{new Date(res.time).toLocaleTimeString()}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
