import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, 
  RefreshCcw, 
  Search, 
  PenTool, 
  BarChart3, 
  Globe, 
  Terminal,
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:8000';

function App() {
  const [topic, setTopic] = useState('Generative AI for Businesses');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const logEndRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/status`);
      setStatus(response.data);
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [status?.logs]);

  const startPipeline = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post(`${API_BASE}/start?topic=${encodeURIComponent(topic)}`);
      fetchStatus();
    } catch (err) {
      setError("Failed to start pipeline. Make sure backend is running.");
      console.error(err);
    }
    setLoading(false);
  };

  const resetPipeline = async () => {
    try {
      await axios.get(`${API_BASE}/reset`);
      fetchStatus();
    } catch (err) {
      console.error(err);
    }
  };

  const agentSteps = [
    { id: 'keyword_research', name: 'Keyword Research', icon: <Search size={20} />, description: 'Analyzing market trends and search volumes.' },
    { id: 'save_blog', name: 'Content Generation', icon: <PenTool size={20} />, description: 'Drafting high-quality blog content based on keywords.' },
    { id: 'seo_optimizer', name: 'SEO Optimization', icon: <BarChart3 size={20} />, description: 'Optimizing meta tags, density, and readability.' },
    { id: 'publish_content', name: 'Distribution', icon: <Globe size={20} />, description: 'Finalizing and publishing content to the platform.' },
  ];

  const isRunning = status?.execution?.is_running;

  return (
    <div className="app-container">
      <header>
        <div className="logo">CONTENT AI AGENT</div>
        <div className="status-badge">
          <div className={`status-dot ${isRunning ? 'active' : ''}`}></div>
          {isRunning ? 'System Processing' : 'System Ready'}
        </div>
      </header>

      <main className="main-grid">
        <div className="left-column">
          <div className="dashboard-card">
            <div className="card-title">
              <Play size={24} className="text-blue-500" />
              Pipeline Configuration
            </div>
            <div className="input-group">
              <input 
                type="text" 
                value={topic} 
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter topic for content creation..."
                disabled={isRunning}
              />
              <button onClick={startPipeline} disabled={isRunning || !topic}>
                {loading ? <Loader2 className="animate-spin" /> : <Play size={18} />}
                Start Pipeline
              </button>
              <button 
                onClick={resetPipeline} 
                className="bg-zinc-800 hover:bg-zinc-700"
                disabled={isRunning}
              >
                <RefreshCcw size={18} />
              </button>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-500 flex items-center gap-2">
                <AlertCircle size={18} />
                {error}
              </div>
            )}

            <div className="steps-container">
              {agentSteps.map((step, index) => {
                const agentData = status?.agents[step.id];
                const isCompleted = agentData?.status === 'completed';
                const isCurrent = agentData?.status === 'started_running' || agentData?.status === 'running';
                
                return (
                  <motion.div 
                    key={step.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`step-card ${isCurrent ? 'active' : ''}`}
                  >
                    <div className="step-header">
                      <div className="step-info">
                        <div className={`step-icon ${isCompleted ? 'text-green-500' : isCurrent ? 'text-blue-500' : 'text-zinc-500'}`}>
                          {isCompleted ? <CheckCircle2 size={24} /> : step.icon}
                        </div>
                        <div>
                          <div className="step-name">{step.name}</div>
                          <div className="text-xs text-zinc-500">{step.description}</div>
                        </div>
                      </div>
                      <div className="step-status">
                        {isCompleted ? (
                          <span className="text-green-500">{agentData.time_taken || 'Done'}</span>
                        ) : isCurrent ? (
                          <span className="text-blue-500 flex items-center gap-1">
                            <Loader2 size={12} className="animate-spin" />
                            Active
                          </span>
                        ) : (
                          <span className="text-zinc-600">Pending</span>
                        )}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            <AnimatePresence>
              {status?.execution?.result && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="result-overlay"
                >
                  <div className="result-title">
                    <CheckCircle2 className="text-green-500" />
                    Final Output
                  </div>
                  <div className="result-body">
                    {status.execution.result}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="right-column">
          <div className="logs-card">
            <div className="p-6 border-b border-zinc-800 flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold">
                <Terminal size={20} />
                Execution Logs
              </div>
              <div className="text-xs text-zinc-500">
                {status?.logs?.length || 0} entries
              </div>
            </div>
            <div className="logs-container">
              {status?.logs?.map((log, i) => (
                <div key={i} className="log-entry">
                  <span className="log-time">[{new Date().toLocaleTimeString()}]</span>
                  {log}
                </div>
              ))}
              {status?.logs?.length === 0 && (
                <div className="text-zinc-600 italic">Waiting for pipeline to start...</div>
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
