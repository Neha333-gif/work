import { useEffect, useState } from 'react';
import { Shield, Lock, AlertCircle, CheckCircle, BarChart3, TrendingUp, Zap, Clock } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { motion } from 'framer-motion';

const SecurityPage = () => {
  const { user } = useAuth();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timeInterval);
  }, []);

  const securityStatus = [
    { label: 'Network Encryption', status: 'active', icon: Lock },
    { label: 'Firewall', status: 'active', icon: Shield },
    { label: 'Two-Factor Auth', status: 'active', icon: CheckCircle },
    { label: 'Intrusion Detection', status: 'active', icon: AlertCircle },
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
          <div className="nav-item">
            <BarChart3 size={20} />
            <span>Dashboard</span>
          </div>
          <div className="nav-item">
            <TrendingUp size={20} />
            <span>Analytics</span>
          </div>
          <div className="nav-item active">
            <Shield size={20} />
            <span>Security</span>
          </div>
          <div className="nav-item">
            <Clock size={20} />
            <span>Settings</span>
          </div>
        </nav>

        <div className="system-card">
          <h4>Security Status</h4>
          <div className="stat-row">
            <span>Overall</span>
            <span style={{ color: 'var(--success)' }}>Secure</span>
          </div>
          <div className="stat-row">
            <span>Alerts</span>
            <span>0 Active</span>
          </div>
          <div className="stat-row">
            <span>Last Scan</span>
            <span>2 min ago</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div className="dashboard-header">
            <h2>Security Center</h2>
            <p style={{ color: 'var(--text-muted)' }}>Monitor and manage system security.</p>
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

        <div className="security-grid">
          {securityStatus.map((item, idx) => {
            const IconComponent = item.icon;
            return (
              <motion.div
                key={idx}
                className="security-card"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: idx * 0.1 }}
              >
                <div className="security-header">
                  <IconComponent size={24} color="var(--primary)" />
                  <span className="security-badge active">Active</span>
                </div>
                <h3>{item.label}</h3>
                <p className="security-desc">Protected and operational</p>
              </motion.div>
            );
          })}
        </div>

        <motion.div
          className="security-alert"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
        >
          <div className="alert-header">
            <CheckCircle size={20} color="var(--success)" />
            <h4>System Status: All Clear</h4>
          </div>
          <p>All security protocols are functioning normally. No threats detected.</p>
        </motion.div>
      </main>
    </div>
  );
};

export default SecurityPage;
