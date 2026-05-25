import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

const AuthContext = createContext({
  user: null,
  token: null,
  login: () => {},
  logout: () => {},
  register: () => {},
  isAuthenticated: false,
});

const createSessionToken = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  useEffect(() => {
    const rawAuth = localStorage.getItem('auraAuth');
    if (!rawAuth) return;

    try {
      const parsed = JSON.parse(rawAuth);
      if (parsed?.user && parsed?.token) {
        setUser(parsed.user);
        setToken(parsed.token);
      }
    } catch (error) {
      localStorage.removeItem('auraAuth');
    }
  }, []);

  const createSession = (userPayload) => {
    const authState = {
      user: {
        username: userPayload.username,
        email: userPayload.email,
      },
      token: createSessionToken(),
    };

    localStorage.setItem('auraAuth', JSON.stringify(authState));
    setUser(authState.user);
    setToken(authState.token);
  };

  const login = ({ email, password }) => {
    const users = JSON.parse(localStorage.getItem('auraUsers') || '[]');
    const match = users.find((userRecord) => userRecord.email === email && userRecord.password === password);
    if (!match) {
      throw new Error('Invalid email or password');
    }

    createSession(match);
  };

  const register = ({ username, email, password }) => {
    const users = JSON.parse(localStorage.getItem('auraUsers') || '[]');
    if (users.some((userRecord) => userRecord.email === email)) {
      throw new Error('A user with this email already exists');
    }

    const newUser = { username, email, password };
    users.push(newUser);
    localStorage.setItem('auraUsers', JSON.stringify(users));
    createSession(newUser);
  };

  const logout = () => {
    localStorage.removeItem('auraAuth');
    setUser(null);
    setToken(null);
  };

  const value = useMemo(
    () => ({
      user,
      token,
      login,
      logout,
      register,
      isAuthenticated: Boolean(token),
    }),
    [user, token]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);
