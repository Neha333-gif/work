import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { UserPlus, Shield } from 'lucide-react';
import { useAuth } from '../AuthContext';

const RegisterPage = () => {
  const { register, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (!username.trim() || !email.trim() || !password.trim()) {
      setError('Please complete all fields.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    try {
      register({ username: username.trim(), email: email.trim().toLowerCase(), password: password.trim() });
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
            <Shield size={20} />
          </div>
          <div>
            <h1>Aura Home OS</h1>
            <p>Register to unlock your smart home dashboard.</p>
          </div>
        </div>

        <div className="auth-hero">
          <h2>Get started.</h2>
          <p>Create a secure account and manage your connected devices from one central hub.</p>
        </div>
      </aside>

      <main className="auth-main">
        <div className="auth-card">
          <div className="auth-header">
            <h2>Create account</h2>
            <p>Registration is quick and secure.</p>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            <label>
              Username
              <input
                type="text"
                placeholder="Your name"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
              />
            </label>

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

            <label>
              Confirm password
              <input
                type="password"
                placeholder="Re-enter password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                required
              />
            </label>

            {error && <p className="auth-error">{error}</p>}

            <button type="submit" className="primary-btn">
              Register <UserPlus size={18} />
            </button>
          </form>

          <div className="auth-footer">
            <p>
              Already have an account? <Link to="/login">Sign in</Link>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default RegisterPage;
