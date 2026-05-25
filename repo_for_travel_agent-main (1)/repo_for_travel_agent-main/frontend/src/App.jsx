import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    name: '',
    destination: '',
    travel_type: 'budget',
    food_type: 'veg',
    comfort: 'budget'
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate plan');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="hero">
        <h1>Antigravity Travel</h1>
        <p>Your AI-Powered Journey Starts Here</p>
      </header>

      <main className="planner-grid">
        <section className="glass-card">
          <h2>Plan Your Trip</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Full Name</label>
              <input 
                type="text" 
                name="name" 
                value={formData.name} 
                onChange={handleChange} 
                placeholder="John Doe" 
                required 
              />
            </div>
            <div className="form-group">
              <label>Destination</label>
              <input 
                type="text" 
                name="destination" 
                value={formData.destination} 
                onChange={handleChange} 
                placeholder="Paris, France" 
                required 
              />
            </div>
            <div className="form-group">
              <label>Travel Style</label>
              <select name="travel_type" value={formData.travel_type} onChange={handleChange}>
                <option value="budget">Budget Friendly</option>
                <option value="comfort">Comfortable</option>
                <option value="luxury">Luxury Elite</option>
              </select>
            </div>
            <div className="form-group">
              <label>Food Preference</label>
              <select name="food_type" value={formData.food_type} onChange={handleChange}>
                <option value="veg">Vegetarian</option>
                <option value="non-veg">Non-Vegetarian</option>
              </select>
            </div>
            <div className="form-group">
              <label>Comfort Level</label>
              <select name="comfort" value={formData.comfort} onChange={handleChange}>
                <option value="budget">Basic</option>
                <option value="comfort">Standard</option>
                <option value="luxury">Premium</option>
              </select>
            </div>
            <button type="submit" disabled={loading}>
              {loading ? 'Crafting Itinerary...' : 'Generate Plan'}
            </button>
          </form>
        </section>

        <section className="results-section">
          {loading && (
            <div className="glass-card" style={{textAlign: 'center'}}>
              <h3>Magical Agents at Work</h3>
              <div className="loading-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
              <p style={{color: 'var(--text-muted)'}}>Connecting with Flight & Hotel specialists...</p>
            </div>
          )}

          {error && (
            <div className="glass-card" style={{borderColor: 'var(--danger)'}}>
              <h3 style={{color: 'var(--danger)'}}>Error</h3>
              <p>{error}</p>
              <p style={{marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)'}}>
                Tip: Check backend logs for more details.
              </p>
            </div>
          )}

          {result && (
            <div className="glass-card">
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
                <h2 style={{margin: 0}}>Your Personalized Itinerary</h2>
                {result.status === 'demo_mode' && (
                  <span style={{background: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b', padding: '0.25rem 0.75rem', borderRadius: '1rem', fontSize: '0.75rem', fontWeight: 'bold'}}>
                    DEMO MODE
                  </span>
                )}
              </div>
              
              {result.warning && (
                <div style={{background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', padding: '1rem', borderRadius: '0.75rem', marginBottom: '1.5rem', color: '#f87171', fontSize: '0.9rem'}}>
                  <strong>Notice:</strong> {result.warning}
                </div>
              )}

              <div style={{whiteSpace: 'pre-wrap', lineHeight: '1.6'}}>
                {result.result}
              </div>
              
              <div style={{marginTop: '2rem'}}>
                <h3>System Logs</h3>
                <div className="history-item">
                  {JSON.stringify(result.history, null, 2)}
                </div>
              </div>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="glass-card" style={{opacity: 0.7, textAlign: 'center'}}>
              <p>Fill out the form to start planning your dream vacation.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
