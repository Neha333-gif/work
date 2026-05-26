import { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart3, TrendingUp, Zap, Clock } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { motion } from 'framer-motion';

const API_BASE = 'http://127.0.0.1:8000';

const AnalyticsPage = () => {
  const { user } = useAuth();
  const [state, setState] = useState({});
  const [currentTime, setCurrentTime] = useState(new Date());

  const fetchState = async () => {
    try {
      const res = await axios.get(`${API_BASE}/state`);
      setState(res.data);
    } catch (error) {
      console.error('Unable to reach backend', error);
    }
  };

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 5000);
    const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => {
      clearInterval(interval);
      clearInterval(timeInterval);
    };
  }, []);

  const activeDevices = Object.values(state).filter((value) => value === true || value?.status).length;
  const totalDevices = Object.keys(state).length || 6;
  const energyConsumption = (activeDevices * 0.2).toFixed(2);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <BarChart3 size={24} color="white" />
          </div>
          <h1>Aura OS</h1>
        </div>

        <nav className="nav-section">
          <div className="nav-item">
            <BarChart3 size={20} />
            <span>Dashboard</span>
          </div>
          <div className="nav-item active">
            <TrendingUp size={20} />
            <span>Analytics</span>
          </div>
          <div className="nav-item">
            <Zap size={20} />
            <span>Security</span>
          </div>
          <div className="nav-item">
            <Clock size={20} />
            <span>Settings</span>
          </div>
        </nav>

        <div className="system-card">
          <h4>Energy Stats</h4>
          <div className="stat-row">
            <span>Current</span>
            <span style={{ color: 'var(--success)' }}>{energyConsumption} kW</span>
          </div>
          <div className="stat-row">
            <span>Active</span>
            <span>{activeDevices} / {totalDevices}</span>
          </div>
          <div className="stat-row">
            <span>Peak Today</span>
            <span>2.4 kW</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div className="dashboard-header">
            <h2>Analytics Dashboard</h2>
            <p style={{ color: 'var(--text-muted)' }}>Monitor your home energy and device usage.</p>
          </div>

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div className="user-profile" style={{ background: 'transparent', border: 'none', padding: 0 }}>
              <div style={{ textAlign: 'right', marginRight: '1rem' }}>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{currentTime.toLocaleDateString()}</div>
              </div>
            </div>
            <div className="user-profile">
              <div className="avatar"></div>
              <span style={{ fontWeight: 600 }}>{user?.username || 'User'}</span>
            </div>
          </div>
        </header>

        <div className="analytics-grid">
          <motion.div
            className="analytics-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="card-header">
              <h3>Energy Consumption</h3>
              <Zap size={20} color="var(--secondary)" />
            </div>
            <div className="card-value">{energyConsumption} kW</div>
            <p className="card-label">Current usage</p>
          </motion.div>

          <motion.div
            className="analytics-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="card-header">
              <h3>Active Devices</h3>
              <TrendingUp size={20} color="var(--primary)" />
            </div>
            <div className="card-value">{activeDevices}</div>
            <p className="card-label">Out of {totalDevices} total</p>
          </motion.div>

          <motion.div
            className="analytics-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="card-header">
              <h3>Uptime</h3>
              <Clock size={20} color="var(--success)" />
            </div>
            <div className="card-value">12d 4h</div>
            <p className="card-label">System operational</p>
          </motion.div>

          <motion.div
            className="analytics-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <div className="card-header">
              <h3>Peak Usage</h3>
              <BarChart3 size={20} color="var(--secondary)" />
            </div>
            <div className="card-value">2.4 kW</div>
            <p className="card-label">Today at 18:30</p>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default AnalyticsPage;
