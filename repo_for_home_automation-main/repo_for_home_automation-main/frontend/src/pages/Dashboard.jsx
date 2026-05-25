import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Lightbulb, Wind, Thermometer, Waves, Heater, Shield, LayoutDashboard, Activity, Settings } from 'lucide-react';
import { useAuth } from '../AuthContext';

const API_BASE = 'http://127.0.0.1:8000';

const DeviceCard = ({ name, data, icon: Icon }) => {
  const isActive = typeof data === 'boolean' ? data : data?.status;
  const preference = data?.user_preference || null;

  const getSpinClass = () => {
    if (!isActive || name !== 'fan') return '';
    return `fan-spin-${preference || 'medium'}`;
  };

  return (
    <motion.div
      layout
      className={`device-card ${isActive ? 'active' : ''} ${preference ? `intensity-${preference}` : ''}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ y: -8 }}
      transition={{ duration: 0.25 }}
    >
      <div className="card-top">
        <div className="icon-container">
          <Icon size={26} className={getSpinClass()} />
        </div>
      </div>

      <div className="device-info">
        <h3>{name.charAt(0).toUpperCase() + name.slice(1)}</h3>
        <p>{isActive ? 'Running smoothly' : 'Currently inactive'}</p>
      </div>

      <div className="intensity-meter">
        <div className="meter-bar bar-1"></div>
        <div className="meter-bar bar-2"></div>
        <div className="meter-bar bar-3"></div>
      </div>

      {preference && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="preference-label">
          MODE: {preference.toUpperCase()}
        </motion.div>
      )}
    </motion.div>
  );
};

const DashboardPage = () => {
  const { user, logout } = useAuth();
  const [state, setState] = useState({});
  const [command, setCommand] = useState('');
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState('');
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
    const stateInterval = setInterval(fetchState, 4000);
    const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => {
      clearInterval(stateInterval);
      clearInterval(timeInterval);
    };
  }, []);

  const handleSendCommand = async (event) => {
    event.preventDefault();
    if (!command.trim()) return;

    setLoading(true);
    setFeedback('AI is processing your request...');

    try {
      const res = await axios.post(`${API_BASE}/command`, { command });
      setFeedback(res.data.result);
      setState(res.data.state);
      setCommand('');
    } catch (error) {
      setFeedback('System Error: Interface connection lost.');
    } finally {
      setLoading(false);
    }
  };

  const devices = [
    { name: 'lights', icon: Lightbulb },
    { name: 'fan', icon: Wind },
    { name: 'ac', icon: Thermometer },
    { name: 'geyser', icon: Waves },
    { name: 'heater', icon: Heater },
    { name: 'robot', icon: Zap },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={24} color="white" />
          </div>
          <h1>Aura OS</h1>
        </div>

        <nav className="nav-section">
          <Link to="/dashboard" className="nav-item active">
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </Link>
          <Link to="/analytics" className="nav-item">
            <Activity size={20} />
            <span>Analytics</span>
          </Link>
          <Link to="/security" className="nav-item">
            <Shield size={20} />
            <span>Security</span>
          </Link>
          <Link to="/settings" className="nav-item">
            <Settings size={20} />
            <span>Settings</span>
          </Link>
        </nav>

        <div className="system-card">
          <h4>System Health</h4>
          <div className="stat-row">
            <span>Power</span>
            <span style={{ color: 'var(--success)' }}>1.2 kW</span>
          </div>
          <div className="stat-row">
            <span>Uptime</span>
            <span>12d 4h</span>
          </div>
          <div className="stat-row">
            <span>Devices</span>
            <span>{Object.values(state).filter((value) => value === true || value?.status).length} / 6</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div className="dashboard-header">
            <h2>Welcome home, {user?.username || 'User'}</h2>
            <p style={{ color: 'var(--text-muted)' }}>All systems are nominal.</p>
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
              <div>
                <span style={{ fontWeight: 600 }}>{user?.username || 'Aura User'}</span>
                <button type="button" className="link-button" onClick={logout}>
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </header>

        <div className="dashboard-grid">
          {devices.map((device) => (
            <DeviceCard key={device.name} name={device.name} data={state[device.name] || false} icon={device.icon} />
          ))}
        </div>

        <section className="command-center">
          <div className="command-heading">
            <Zap size={20} color="var(--primary)" />
            <h3>Neural Command Link</h3>
          </div>

          <form onSubmit={handleSendCommand} className="command-form">
            <div className="input-box">
              <input
                type="text"
                placeholder="Type a command... (e.g. 'Boost the AC and turn off lights')"
                value={command}
                onChange={(event) => setCommand(event.target.value)}
                disabled={loading}
              />
              <button type="submit" className="send-btn" disabled={loading || !command.trim()}>
                {loading ? 'Sending...' : <Zap size={22} />}
              </button>
            </div>
          </form>

          <AnimatePresence>
            {feedback && (
              <motion.div className="feedback-bubble" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <div className="pulse"></div>
                  <p style={{ lineHeight: 1.6 }}>{feedback}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>
    </div>
  );
};

export default DashboardPage;
