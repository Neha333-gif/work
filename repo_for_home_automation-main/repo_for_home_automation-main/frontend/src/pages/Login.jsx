import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Lock, ArrowRight } from 'lucide-react';
import { useAuth } from '../AuthContext';

const LoginPage = () => {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    try {
      login({ email: email.trim().toLowerCase(), password: password.trim() });
      navigate('/dashboard');
    } catch (authError) {
      setError(authError.message);
    }
  };

  return (
    <div className="auth-shell">
      <aside className="auth-panel">
        <div className="brand-mark">
          <div className="brand-icon-small">
            <Lock size={20} />
          </div>
          <div>
            <h1>Aura Home OS</h1>
            <p>Secure access to your smart home dashboard.</p>
          </div>
        </div>

        <div className="auth-hero">
          <h2>Welcome back.</h2>
          <p>Log in to manage devices, monitor energy, and control your home with intelligence.</p>
        </div>
      </aside>

      <main className="auth-main">
        <div className="auth-card">
          <div className="auth-header">
            <h2>Sign in</h2>
            <p>Enter your credentials to continue.</p>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            <label>
              Email address
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </label>

            <label>
              Password
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </label>

            {error && <p className="auth-error">{error}</p>}

            <button type="submit" className="primary-btn">
              Continue <ArrowRight size={18} />
            </button>
          </form>

          <div className="auth-footer">
            <p>
              New to Aura? <Link to="/register">Create an account</Link>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default LoginPage;
