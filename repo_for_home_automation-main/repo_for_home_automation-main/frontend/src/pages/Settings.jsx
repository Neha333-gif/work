import { useEffect, useState } from 'react';
import { Settings, Bell, Eye, Moon, Volume2, BarChart3, TrendingUp, Zap, Clock } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { motion } from 'framer-motion';

const SettingsPage = () => {
  const { user, logout } = useAuth();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [settings, setSettings] = useState({
    notifications: true,
    darkMode: true,
    soundAlerts: true,
    autoUpdates: true,
  });

  useEffect(() => {
    const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timeInterval);
  }, []);

  const toggleSetting = (key) => {
    setSettings((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Settings size={24} color="white" />
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
          <div className="nav-item">
            <Zap size={20} />
            <span>Security</span>
          </div>
          <div className="nav-item active">
            <Clock size={20} />
            <span>Settings</span>
          </div>
        </nav>

        <div className="system-card">
          <h4>Preferences</h4>
          <div className="stat-row">
            <span>Theme</span>
            <span>{settings.darkMode ? 'Dark' : 'Light'}</span>
          </div>
          <div className="stat-row">
            <span>Notifications</span>
            <span>{settings.notifications ? 'On' : 'Off'}</span>
          </div>
          <div className="stat-row">
            <span>Sound Alerts</span>
            <span>{settings.soundAlerts ? 'On' : 'Off'}</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div className="dashboard-header">
            <h2>Settings</h2>
            <p style={{ color: 'var(--text-muted)' }}>Manage your preferences and account.</p>
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

        <div className="settings-container">
          <motion.section
            className="settings-section"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h3>Account</h3>
            <div className="settings-item">
              <div className="item-label">
                <span className="item-title">Username</span>
                <p className="item-desc">Your account name</p>
              </div>
              <span className="item-value">{user?.username}</span>
            </div>
            <div className="settings-item">
              <div className="item-label">
                <span className="item-title">Email</span>
                <p className="item-desc">Account email address</p>
              </div>
              <span className="item-value">{user?.email}</span>
            </div>
          </motion.section>

          <motion.section
            className="settings-section"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <h3>Preferences</h3>
            <div className="settings-toggle">
              <div className="toggle-label">
                <Bell size={20} />
                <div>
                  <span className="item-title">Notifications</span>
                  <p className="item-desc">Receive system alerts</p>
                </div>
              </div>
              <button
                className={`toggle-btn ${settings.notifications ? 'active' : ''}`}
                onClick={() => toggleSetting('notifications')}
              >
                <div className="toggle-knob"></div>
              </button>
            </div>

            <div className="settings-toggle">
              <div className="toggle-label">
                <Moon size={20} />
                <div>
                  <span className="item-title">Dark Mode</span>
                  <p className="item-desc">Use dark theme</p>
                </div>
              </div>
              <button
                className={`toggle-btn ${settings.darkMode ? 'active' : ''}`}
                onClick={() => toggleSetting('darkMode')}
              >
                <div className="toggle-knob"></div>
              </button>
            </div>

            <div className="settings-toggle">
              <div className="toggle-label">
                <Volume2 size={20} />
                <div>
                  <span className="item-title">Sound Alerts</span>
                  <p className="item-desc">Audio notifications</p>
                </div>
              </div>
              <button
                className={`toggle-btn ${settings.soundAlerts ? 'active' : ''}`}
                onClick={() => toggleSetting('soundAlerts')}
              >
                <div className="toggle-knob"></div>
              </button>
            </div>
          </motion.section>

          <motion.section
            className="settings-section"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <h3>Danger Zone</h3>
            <button className="btn-danger" onClick={logout}>
              Sign out
            </button>
          </motion.section>
        </div>
      </main>
    </div>
  );
};

export default SettingsPage;
