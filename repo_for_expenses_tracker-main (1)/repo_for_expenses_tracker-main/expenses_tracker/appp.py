from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
import traceback
from expenses_tracker_2 import run_tracker

app = Flask(__name__)
app.secret_key = "finance_tracker_secret_2024"

# In-memory user store (demo)
users = {}

# ──────────────────────────────────────────────────────────────
# SHARED CSS / FONTS
# ──────────────────────────────────────────────────────────────
BASE_ASSETS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --void:       #070a0f;
  --deep:       #0c1118;
  --surface:    #111920;
  --raised:     #172130;
  --border:     rgba(255,255,255,0.06);
  --border2:    rgba(255,255,255,0.12);
  --gold:       #d4a843;
  --gold2:      #f0c860;
  --gold-dim:   rgba(212,168,67,0.15);
  --teal:       #2dd4bf;
  --teal-dim:   rgba(45,212,191,0.12);
  --rose:       #fb7185;
  --rose-dim:   rgba(251,113,133,0.12);
  --violet:     #a78bfa;
  --text:       #e8edf5;
  --text2:      #9ba8bb;
  --text3:      #5a6a80;
  --success:    #34d399;
  --warn:       #fbbf24;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--void);
  color: var(--text);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--gold); text-decoration: none; transition: color 0.2s; }
a:hover { color: var(--gold2); }
::selection { background: var(--gold-dim); color: var(--gold2); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--raised); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border2); }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(212,168,67,0.03) 0%, transparent 60%);
  pointer-events: none;
}
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem;
  padding: 0.8rem 1.6rem; border: none; border-radius: 12px;
  font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700;
  cursor: pointer; transition: all 0.25s; letter-spacing: 0.3px;
  white-space: nowrap;
}
.btn-gold {
  background: linear-gradient(135deg, var(--gold) 0%, #b8902e 100%);
  color: var(--void);
  box-shadow: 0 4px 20px rgba(212,168,67,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
}
.btn-gold:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(212,168,67,0.45), inset 0 1px 0 rgba(255,255,255,0.2);
}
.btn-gold:active { transform: translateY(0); }
.btn-ghost {
  background: transparent;
  border: 1px solid var(--border2);
  color: var(--text2);
}
.btn-ghost:hover { border-color: var(--gold); color: var(--gold); background: var(--gold-dim); }
.form-group { display: flex; flex-direction: column; gap: 0.5rem; }
.form-label {
  font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.8px; color: var(--text3);
  font-family: 'Syne', sans-serif;
}
.form-input {
  padding: 0.85rem 1rem; border-radius: 10px;
  border: 1px solid var(--border); background: var(--deep);
  color: var(--text); font-size: 0.95rem; outline: none;
  font-family: 'DM Sans', sans-serif;
  transition: all 0.2s;
}
.form-input:focus { border-color: var(--gold); box-shadow: 0 0 0 3px rgba(212,168,67,0.1); }
.form-input::placeholder { color: var(--text3); }
.tag {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.25rem 0.7rem; border-radius: 99px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.4px;
  font-family: 'Syne', sans-serif;
}
.tag-gold  { background: var(--gold-dim);  color: var(--gold2);  border: 1px solid rgba(212,168,67,0.25); }
.tag-teal  { background: var(--teal-dim);  color: var(--teal);   border: 1px solid rgba(45,212,191,0.2); }
.tag-rose  { background: var(--rose-dim);  color: var(--rose);   border: 1px solid rgba(251,113,133,0.2); }
.tag-violet{ background: rgba(167,139,250,0.1); color: var(--violet); border: 1px solid rgba(167,139,250,0.2); }
.flash-error {
  background: var(--rose-dim); border: 1px solid rgba(251,113,133,0.25);
  color: var(--rose); border-radius: 10px; padding: 0.85rem 1.1rem;
  font-size: 0.875rem;
}
</style>
"""

# ──────────────────────────────────────────────────────────────
# LOGIN PAGE
# ──────────────────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign In — Vaultex AI</title>
  """ + BASE_ASSETS + """
  <style>
    body {
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh;
      background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(212,168,67,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 80% 80%, rgba(45,212,191,0.04) 0%, transparent 50%),
        var(--void);
    }
    .auth-wrap { position: relative; z-index: 1; width: 100%; max-width: 440px; padding: 1rem; }
    .auth-box { padding: 2.8rem 2.4rem; animation: fadeUp 0.5s ease both; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
    .logo { text-align:center; margin-bottom:2.2rem; }
    .logo-mark {
      width:56px; height:56px; margin:0 auto 1rem;
      background: linear-gradient(135deg, var(--gold), #8a5c10);
      border-radius:16px; display:flex; align-items:center; justify-content:center;
      font-size:1.5rem; box-shadow:0 8px 32px rgba(212,168,67,0.3);
    }
    .logo h1 { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; letter-spacing:-0.5px; color:var(--text); }
    .logo h1 span { color:var(--gold); }
    .logo p { color:var(--text3); font-size:0.88rem; margin-top:0.3rem; }
    form { display:flex; flex-direction:column; gap:1.1rem; }
    .form-footer { text-align:center; color:var(--text3); font-size:0.85rem; margin-top:1rem; }
  </style>
</head>
<body>
  <div class="auth-wrap">
    <div class="auth-box card">
      <div class="logo">
        <div class="logo-mark">&#x1F4B9;</div>
        <h1>Vault<span>ex</span></h1>
        <p>AI-powered personal finance intelligence</p>
      </div>
      {% if error %}
      <div class="flash-error" style="margin-bottom:1.25rem">{{ error }}</div>
      {% endif %}
      <form method="POST" action="/login">
        <div class="form-group">
          <label class="form-label">Username</label>
          <input class="form-input" type="text" name="username" placeholder="your_username" required autofocus autocomplete="username">
        </div>
        <div class="form-group">
          <label class="form-label">Password</label>
          <input class="form-input" type="password" name="password" placeholder="Enter password" required autocomplete="current-password">
        </div>
        <button class="btn btn-gold" type="submit" style="width:100%;margin-top:0.4rem;padding:1rem">
          Sign In &rarr;
        </button>
      </form>
      <div class="form-footer" style="margin-top:1.5rem">No account? <a href="/register">Create one</a></div>
    </div>
  </div>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────
# REGISTER PAGE
# ──────────────────────────────────────────────────────────────
REGISTER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Register — Vaultex AI</title>
  """ + BASE_ASSETS + """
  <style>
    body {
      display:flex; align-items:center; justify-content:center; min-height:100vh;
      background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(45,212,191,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 20% 80%, rgba(212,168,67,0.04) 0%, transparent 50%),
        var(--void);
    }
    .auth-wrap { position:relative; z-index:1; width:100%; max-width:460px; padding:1rem; }
    .auth-box { padding:2.8rem 2.4rem; animation:fadeUp 0.5s ease both; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
    .logo { text-align:center; margin-bottom:2rem; }
    .logo-mark {
      width:56px; height:56px; margin:0 auto 1rem;
      background:linear-gradient(135deg,var(--teal),#0d8a7a);
      border-radius:16px; display:flex; align-items:center; justify-content:center;
      font-size:1.5rem; box-shadow:0 8px 32px rgba(45,212,191,0.25);
    }
    .logo h1 { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:var(--text); }
    .logo p { color:var(--text3); font-size:0.88rem; margin-top:0.3rem; }
    form { display:flex; flex-direction:column; gap:1rem; }
    .row-2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
    .form-footer { text-align:center; color:var(--text3); font-size:0.85rem; margin-top:1rem; }
    @media(max-width:400px) { .row-2 { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="auth-wrap">
    <div class="auth-box card">
      <div class="logo">
        <div class="logo-mark">&#x1F680;</div>
        <h1>Create Account</h1>
        <p>Start tracking your finances with AI</p>
      </div>
      {% if error %}
      <div class="flash-error" style="margin-bottom:1rem">{{ error }}</div>
      {% endif %}
      <form method="POST" action="/register">
        <div class="form-group">
          <label class="form-label">Full Name</label>
          <input class="form-input" type="text" name="fullname" placeholder="Jane Smith" required autofocus>
        </div>
        <div class="row-2">
          <div class="form-group">
            <label class="form-label">Username</label>
            <input class="form-input" type="text" name="username" placeholder="jane_smith" required>
          </div>
          <div class="form-group">
            <label class="form-label">Password</label>
            <input class="form-input" type="password" name="password" placeholder="Password" required>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Confirm Password</label>
          <input class="form-input" type="password" name="confirm" placeholder="Repeat password" required>
        </div>
        <button class="btn btn-gold" type="submit" style="width:100%;padding:1rem;margin-top:0.3rem">
          Create Account &rarr;
        </button>
      </form>
      <div class="form-footer" style="margin-top:1.5rem">Already have an account? <a href="/login">Sign in</a></div>
    </div>
  </div>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ──────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard — Vaultex AI</title>
  """ + BASE_ASSETS + """
  <style>
    body {
      background:
        radial-gradient(ellipse 100% 40% at 50% 0%, rgba(212,168,67,0.06) 0%, transparent 55%),
        var(--void);
    }
    .nav {
      position:sticky; top:0; z-index:100;
      display:flex; align-items:center; justify-content:space-between;
      padding:0 2.5rem; height:64px;
      background:rgba(7,10,15,0.85);
      border-bottom:1px solid var(--border);
      backdrop-filter:blur(16px);
    }
    .nav-brand {
      display:flex; align-items:center; gap:0.6rem;
      font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
    }
    .nav-brand .mark {
      width:32px; height:32px; background:linear-gradient(135deg,var(--gold),#8a5c10);
      border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.85rem;
    }
    .nav-brand span { color:var(--gold); }
    .nav-right { display:flex; align-items:center; gap:1.2rem; }
    .nav-user {
      display:flex; align-items:center; gap:0.5rem;
      padding:0.35rem 0.75rem; border-radius:8px;
      background:var(--raised); border:1px solid var(--border);
    }
    .nav-avatar {
      width:26px; height:26px; border-radius:50%;
      background:linear-gradient(135deg,var(--gold),var(--teal));
      display:flex; align-items:center; justify-content:center;
      font-size:0.7rem; font-weight:700; color:var(--void);
    }
    .nav-name { font-size:0.85rem; font-weight:500; color:var(--text2); }
    .main { max-width:1200px; margin:0 auto; padding:2.5rem 2rem 5rem; }

    /* HERO */
    .hero { margin-bottom:2.5rem; }
    .hero-eyebrow {
      display:inline-flex; align-items:center; gap:0.5rem;
      padding:0.3rem 0.75rem; border-radius:99px;
      background:var(--gold-dim); border:1px solid rgba(212,168,67,0.2);
      font-size:0.75rem; font-weight:600; color:var(--gold);
      font-family:'Syne',sans-serif; letter-spacing:0.5px; margin-bottom:1rem;
    }
    .hero-eyebrow .pulse {
      width:6px; height:6px; border-radius:50%; background:var(--gold);
      animation:pulse 2s ease infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
    .hero h1 {
      font-family:'Syne',sans-serif;
      font-size:clamp(1.8rem,4vw,2.8rem); font-weight:800; letter-spacing:-1px; line-height:1.1;
    }
    .hero h1 em { font-style:normal; color:var(--gold); }
    .hero p { color:var(--text2); margin-top:0.75rem; font-size:1rem; max-width:520px; line-height:1.6; }

    /* STATS */
    .stats-row {
      display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
      gap:1rem; margin-bottom:2rem;
    }
    .stat-card {
      padding:1.25rem 1.4rem; border-radius:16px;
      background:var(--surface); border:1px solid var(--border); transition:border-color 0.2s;
    }
    .stat-card:hover { border-color:var(--border2); }
    .stat-card .label {
      font-size:0.72rem; text-transform:uppercase; letter-spacing:0.8px;
      color:var(--text3); font-family:'Syne',sans-serif; font-weight:600; margin-bottom:0.5rem;
    }
    .stat-card .value { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; }
    .stat-card .sub { font-size:0.78rem; color:var(--text3); margin-top:0.2rem; }
    .val-gold { color:var(--gold); }
    .val-teal { color:var(--teal); }
    .val-violet { color:var(--violet); }

    /* INPUT PANEL */
    .input-panel { padding:2rem; margin-bottom:2rem; }
    .panel-header { display:flex; align-items:center; gap:0.75rem; margin-bottom:1.6rem; }
    .panel-header h2 { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; }
    .input-row { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-bottom:1.5rem; }
    @media(max-width:600px) { .input-row { grid-template-columns:1fr; } }
    .run-btn { width:100%; padding:1.05rem; font-size:1rem; letter-spacing:0.3px; }

    /* LOADER */
    .loader-overlay { display:none; padding:2rem; text-align:center; }
    .agent-pipeline {
      display:flex; flex-direction:column; gap:0.7rem;
      max-width:420px; margin:1.5rem auto 0; text-align:left;
    }
    .agent-step {
      display:flex; align-items:center; gap:0.85rem;
      padding:0.75rem 1rem; border-radius:10px;
      background:var(--deep); border:1px solid var(--border);
      transition:all 0.4s; opacity:0.4;
    }
    .agent-step.active { opacity:1; border-color:var(--gold); background:var(--gold-dim); }
    .agent-step.done   { opacity:0.7; border-color:rgba(45,212,191,0.2); background:var(--teal-dim); }
    .step-icon { font-size:1.1rem; flex-shrink:0; }
    .step-text strong { font-family:'Syne',sans-serif; font-weight:700; display:block; font-size:0.82rem; color:var(--text); }
    .step-text span { color:var(--text3); font-size:0.78rem; }
    .step-badge { margin-left:auto; flex-shrink:0; font-size:0.7rem; padding:0.15rem 0.5rem; border-radius:4px; }
    .spinner-ring {
      width:44px; height:44px; margin:0 auto; border-radius:50%;
      border:3px solid var(--border); border-top-color:var(--gold);
      animation:spin 0.8s linear infinite;
    }
    @keyframes spin { to{transform:rotate(360deg)} }
    .loader-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; margin-top:1rem; color:var(--text); }
    .loader-sub { font-size:0.83rem; color:var(--text3); margin-top:0.3rem; }

    /* RESULTS */
    #results-section { display:none; }
    .results-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-bottom:1.25rem; }
    @media(max-width:760px) { .results-grid { grid-template-columns:1fr; } }
    .full-width { grid-column:1 / -1; }
    .result-card { padding:1.5rem; }
    .result-card-header {
      display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem;
    }
    .result-card-header h3 { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; display:flex; align-items:center; gap:0.5rem; }
    .result-icon { width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.95rem; }
    .icon-gold   { background:var(--gold-dim); }
    .icon-teal   { background:var(--teal-dim); }
    .icon-rose   { background:var(--rose-dim); }
    .icon-violet { background:rgba(167,139,250,0.12); }

    /* Transaction cards */
    .tx-list { display:flex; flex-direction:column; gap:0.6rem; }
    .tx-item {
      display:flex; align-items:center; gap:0.85rem;
      padding:0.75rem 1rem; border-radius:10px;
      background:var(--deep); border:1px solid var(--border);
      animation:fadeIn 0.4s ease both;
    }
    @keyframes fadeIn { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:translateX(0)} }
    .tx-cat-icon { width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
    .tx-info { flex:1; min-width:0; }
    .tx-cat { font-weight:600; font-size:0.85rem; text-transform:capitalize; }
    .tx-date { font-size:0.75rem; color:var(--text3); margin-top:0.1rem; }
    .tx-amount { font-family:'DM Mono',monospace; font-weight:500; font-size:0.9rem; color:var(--rose); flex-shrink:0; }
    .tx-id { font-size:0.7rem; color:var(--text3); }

    /* Alerts */
    .alert-list { display:flex; flex-direction:column; gap:0.6rem; }
    .alert-item {
      display:flex; align-items:flex-start; gap:0.75rem;
      padding:0.75rem 0.9rem; border-radius:10px;
      font-size:0.85rem; line-height:1.5;
    }
    .alert-warn { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2); color:var(--warn); }
    .alert-ok   { background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.2); color:var(--success); }

    /* Insights */
    .insight-list { display:flex; flex-direction:column; gap:0.6rem; }
    .insight-item {
      display:flex; align-items:flex-start; gap:0.6rem;
      padding:0.75rem 0.9rem; border-radius:10px;
      background:rgba(167,139,250,0.06); border:1px solid rgba(167,139,250,0.12);
      font-size:0.85rem; color:var(--text2); line-height:1.5;
    }
    .insight-dot { width:6px; height:6px; border-radius:50%; background:var(--violet); flex-shrink:0; margin-top:0.45rem; }

    /* Crew output */
    .crew-output { font-size:0.875rem; color:var(--text2); line-height:1.75; white-space:pre-wrap; word-break:break-word; }
    .crew-output strong { color:var(--text); }

    /* Chart */
    .chart-wrap { position:relative; height:220px; }

    /* Error */
    .error-banner {
      padding:1.2rem 1.4rem;
      background:var(--rose-dim); border:1px solid rgba(251,113,133,0.2);
      border-radius:14px; color:var(--rose);
      display:flex; gap:0.75rem; align-items:flex-start;
      margin-bottom:1.5rem;
    }
    .error-banner pre { font-family:'DM Mono',monospace; font-size:0.78rem; white-space:pre-wrap; }

    /* Empty state */
    .empty-state { text-align:center; padding:3rem 2rem; color:var(--text3); }
    .empty-state .big-icon { font-size:2.5rem; margin-bottom:1rem; opacity:0.5; }
    .empty-state p { font-size:0.9rem; }

    /* Section header */
    .section-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem; }
    .section-header h2 { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800; letter-spacing:-0.3px; }
  </style>
</head>
<body>

<nav class="nav">
  <div class="nav-brand">
    <div class="mark">&#x1F4B9;</div>
    Vault<span>ex</span>
  </div>
  <div class="nav-right">
    <div class="nav-user">
      <div class="nav-avatar" id="avatarInitial"></div>
      <span class="nav-name" id="navName">{{ username }}</span>
    </div>
    <form method="POST" action="/logout" style="margin:0">
      <button class="btn btn-ghost" type="submit" style="padding:0.45rem 1rem;font-size:0.82rem">Sign out</button>
    </form>
  </div>
</nav>

<main class="main">
  <div class="hero">
    <div class="hero-eyebrow"><div class="pulse"></div> AI Finance Intelligence</div>
    <h1>Your Money,<br><em>Analyzed in Seconds</em></h1>
    <p>Five specialized AI agents collaborate to decode your spending, detect habits, and recommend your next financial move.</p>
  </div>

  <div class="stats-row">
    <div class="stat-card">
      <div class="label">AI Agents</div>
      <div class="value val-gold">5</div>
      <div class="sub">All active &amp; ready</div>
    </div>
    <div class="stat-card">
      <div class="label">Model</div>
      <div class="value val-teal" style="font-size:1.1rem;padding-top:4px">Llama 3.1</div>
      <div class="sub">via Groq &middot; 8B Instant</div>
    </div>
    <div class="stat-card">
      <div class="label">Analysis</div>
      <div class="value val-violet">Full</div>
      <div class="sub">Transactions &middot; Habits &middot; Budget</div>
    </div>
    <div class="stat-card">
      <div class="label">Status</div>
      <div class="value" style="color:var(--success)">&#9679; Live</div>
      <div class="sub">Real-time simulation</div>
    </div>
  </div>

  <div class="card input-panel">
    <div class="panel-header">
      <div class="result-icon icon-gold">&#x2699;&#xFE0F;</div>
      <h2>Configure Simulation</h2>
      <span class="tag tag-gold" style="margin-left:auto">New Run</span>
    </div>
    <div class="input-row">
      <div class="form-group">
        <label class="form-label">Monthly Income ($)</label>
        <input class="form-input" type="number" id="income" value="50000" min="0" placeholder="50000">
      </div>
      <div class="form-group">
        <label class="form-label">Max Single Expense ($)</label>
        <input class="form-input" type="number" id="max_expense" value="500" min="0" placeholder="500">
      </div>
    </div>
    <button class="btn btn-gold run-btn" id="runBtn" onclick="runAgents()">
      &#x26A1; Launch AI Analysis
    </button>

    <div class="loader-overlay" id="loader">
      <div class="spinner-ring"></div>
      <div class="loader-title">Agents are collaborating&hellip;</div>
      <div class="loader-sub">This may take up to 60 seconds depending on API load</div>
      <div class="agent-pipeline">
        <div class="agent-step" id="step1">
          <span class="step-icon">&#x1F4CA;</span>
          <div class="step-text">
            <strong>Finance Agent</strong>
            <span>Fetching &amp; analyzing transactions</span>
          </div>
          <span class="step-badge tag tag-gold">Active</span>
        </div>
        <div class="agent-step" id="step2">
          <span class="step-icon">&#x1F4B0;</span>
          <div class="step-text">
            <strong>Budget Agent</strong>
            <span>Building personalized budget plan</span>
          </div>
          <span class="step-badge tag" style="opacity:0.4">Waiting</span>
        </div>
        <div class="agent-step" id="step3">
          <span class="step-icon">&#x1F514;</span>
          <div class="step-text">
            <strong>Alert Agent</strong>
            <span>Monitoring for overspending</span>
          </div>
          <span class="step-badge tag" style="opacity:0.4">Waiting</span>
        </div>
        <div class="agent-step" id="step4">
          <span class="step-icon">&#x1F50D;</span>
          <div class="step-text">
            <strong>Habit Agent</strong>
            <span>Detecting purchase patterns</span>
          </div>
          <span class="step-badge tag" style="opacity:0.4">Waiting</span>
        </div>
        <div class="agent-step" id="step5">
          <span class="step-icon">&#x1F4BE;</span>
          <div class="step-text">
            <strong>Memory Agent</strong>
            <span>Saving insights to long-term memory</span>
          </div>
          <span class="step-badge tag" style="opacity:0.4">Waiting</span>
        </div>
      </div>
    </div>
  </div>

  <div class="error-banner" id="errorBanner" style="display:none">
    <span style="font-size:1.1rem;flex-shrink:0">&#x26A0;&#xFE0F;</span>
    <div>
      <strong style="font-family:'Syne',sans-serif;font-size:0.85rem">An error occurred</strong>
      <pre id="errorText" style="margin-top:0.4rem;color:inherit"></pre>
    </div>
  </div>

  <div id="results-section">
    <div class="section-header">
      <h2>Analysis Results</h2>
      <span class="tag tag-teal" id="resultTimestamp"></span>
    </div>
    <div class="results-grid">

      <div class="card result-card full-width">
        <div class="result-card-header">
          <h3><div class="result-icon icon-gold">&#x1F916;</div> AI Crew Final Report</h3>
          <span class="tag tag-gold">CrewAI</span>
        </div>
        <div class="crew-output" id="crewOutput"></div>
      </div>

      <div class="card result-card">
        <div class="result-card-header">
          <h3><div class="result-icon icon-teal">&#x1F9FE;</div> Simulated Transactions</h3>
          <span class="tag tag-teal" id="txCount">0 items</span>
        </div>
        <div class="tx-list" id="txList">
          <div class="empty-state"><div class="big-icon">&#x1F4B3;</div><p>No transactions yet</p></div>
        </div>
      </div>

      <div class="card result-card">
        <div class="result-card-header">
          <h3><div class="result-icon icon-violet">&#x1F4C8;</div> Spending by Category</h3>
          <span class="tag tag-violet">Chart</span>
        </div>
        <div class="chart-wrap"><canvas id="categoryChart"></canvas></div>
      </div>

      <div class="card result-card">
        <div class="result-card-header">
          <h3><div class="result-icon icon-rose">&#x1F514;</div> Alerts</h3>
          <span class="tag tag-rose" id="alertCount">0 alerts</span>
        </div>
        <div class="alert-list" id="alertList">
          <div class="empty-state"><div class="big-icon">&#x1F515;</div><p>No alerts</p></div>
        </div>
      </div>

      <div class="card result-card">
        <div class="result-card-header">
          <h3><div class="result-icon icon-violet">&#x1F4A1;</div> AI Insights</h3>
          <span class="tag tag-violet" id="insightCount">0 insights</span>
        </div>
        <div class="insight-list" id="insightList">
          <div class="empty-state"><div class="big-icon">&#x1F9E0;</div><p>No insights yet</p></div>
        </div>
      </div>

    </div>
  </div>
</main>

<script>
const uname = "{{ username }}";
document.getElementById('navName').textContent = uname;
document.getElementById('avatarInitial').textContent = uname.charAt(0).toUpperCase();

const CAT_ICONS = {
  groceries:'🛒','dining out':'🍽️',transportation:'🚗',
  utilities:'💡',entertainment:'🎬',shopping:'🛍️',health:'🏥',others:'📦'
};
const CAT_COLORS = {
  groceries:'#34d399','dining out':'#f59e0b',transportation:'#60a5fa',
  utilities:'#a78bfa',entertainment:'#f472b6',shopping:'#fb923c',health:'#2dd4bf',others:'#94a3b8'
};

let chartInstance = null;

function animateSteps() {
  const steps = ['step1','step2','step3','step4','step5'];
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
    const b = el.querySelector('.step-badge');
    b.style.opacity = '0.4'; b.textContent = 'Waiting'; b.className = 'step-badge tag';
  });
  const delays = [0, 8000, 18000, 26000, 34000];
  steps.forEach((id,i) => {
    setTimeout(() => {
      if (i > 0) {
        const prev = document.getElementById(steps[i-1]);
        prev.classList.remove('active'); prev.classList.add('done');
        const pb = prev.querySelector('.step-badge');
        pb.textContent = '✓ Done'; pb.className = 'step-badge tag tag-teal'; pb.style.opacity = '1';
      }
      const el = document.getElementById(id); el.classList.add('active');
      const b = el.querySelector('.step-badge');
      b.textContent = 'Active'; b.className = 'step-badge tag tag-gold'; b.style.opacity = '1';
    }, delays[i]);
  });
}

function stopStepAnimation() {
  ['step1','step2','step3','step4','step5'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active'); el.classList.add('done');
    const b = el.querySelector('.step-badge');
    b.textContent = '✓ Done'; b.className = 'step-badge tag tag-teal'; b.style.opacity = '1';
  });
}

async function runAgents() {
  const btn = document.getElementById('runBtn');
  const loader = document.getElementById('loader');
  const results = document.getElementById('results-section');
  const errorBanner = document.getElementById('errorBanner');
  const income = document.getElementById('income').value;
  const maxExp = document.getElementById('max_expense').value;

  btn.style.display = 'none';
  loader.style.display = 'block';
  results.style.display = 'none';
  errorBanner.style.display = 'none';
  animateSteps();

  try {
    const resp = await fetch('/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({income:parseInt(income), max_expense:parseInt(maxExp)})
    });
    const data = await resp.json();
    stopStepAnimation();

    if (data.error) {
      document.getElementById('errorText').textContent = data.error + (data.trace ? '\\n\\n' + data.trace : '');
      errorBanner.style.display = 'flex';
    } else {
      renderResults(data);
      results.style.display = 'block';
      results.scrollIntoView({behavior:'smooth',block:'start'});
    }
  } catch(err) {
    stopStepAnimation();
    document.getElementById('errorText').textContent = err.message;
    errorBanner.style.display = 'flex';
  } finally {
    btn.style.display = 'block';
    loader.style.display = 'none';
  }
}

function renderResults(data) {
  const now = new Date();
  document.getElementById('resultTimestamp').textContent =
    now.toLocaleDateString('en-US',{month:'short',day:'numeric'}) + ' · ' +
    now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});

  document.getElementById('crewOutput').innerHTML = formatCrewOutput(data.results || '');

  const allTx = (data.memory.transactions || []).flat();
  renderTransactions(allTx);
  renderAlerts(data.memory.alerts || []);
  renderInsights(data.memory.insights || []);
  renderChart(allTx);
}

function formatCrewOutput(text) {
  return text
    .replace(/[*][*](.*?)[*][*]/g, '<strong>$1</strong>')
    .split('\\n')
    .map(line => {
      const t = line.trim();
      if (t.startsWith('- ') || t.startsWith('• '))
        return '<div style="display:flex;gap:0.5rem;margin-bottom:0.3rem"><span style="color:var(--gold)">•</span><span>' + t.slice(2) + '</span></div>';
      return '<div>' + line + '</div>';
    }).join('');
}

function renderTransactions(txs) {
  document.getElementById('txCount').textContent = txs.length + ' items';
  const list = document.getElementById('txList');
  if (!txs.length) { list.innerHTML = '<div class="empty-state"><div class="big-icon">💳</div><p>No transactions</p></div>'; return; }
  list.innerHTML = txs.map((t,i) => {
    const icon  = CAT_ICONS[t.category] || '📦';
    const color = CAT_COLORS[t.category] || '#94a3b8';
    return `<div class="tx-item" style="animation-delay:${i*0.07}s">
      <div class="tx-cat-icon" style="background:${color}20;border:1px solid ${color}30">${icon}</div>
      <div class="tx-info">
        <div class="tx-cat">${t.category}</div>
        <div class="tx-date">${t.date} · <span class="tx-id">#${t.id}</span></div>
      </div>
      <div class="tx-amount">-$${t.price}</div>
    </div>`;
  }).join('');
}

function renderAlerts(alerts) {
  document.getElementById('alertCount').textContent = alerts.length + ' alerts';
  const list = document.getElementById('alertList');
  if (!alerts.length) { list.innerHTML = '<div class="empty-state"><div class="big-icon">🔕</div><p>No alerts triggered</p></div>'; return; }
  list.innerHTML = alerts.map(a => {
    const s = String(a).toLowerCase();
    const cls = (s.includes('alert')||s.includes('warn')||s.includes('exceed')||s.includes('more')||s.includes('nearing'))
      ? 'alert-warn' : 'alert-ok';
    const icon = cls === 'alert-warn' ? '⚠️' : '✅';
    return `<div class="alert-item ${cls}">${icon} ${a}</div>`;
  }).join('');
}

function renderInsights(insights) {
  document.getElementById('insightCount').textContent = insights.length + ' insights';
  const list = document.getElementById('insightList');
  const flat = insights.map(i => typeof i === 'object' ? (i.insights || JSON.stringify(i)) : String(i));
  if (!flat.length) { list.innerHTML = '<div class="empty-state"><div class="big-icon">🧠</div><p>No insights yet</p></div>'; return; }
  list.innerHTML = flat.map(i => `<div class="insight-item"><div class="insight-dot"></div>${i}</div>`).join('');
}

function renderChart(txs) {
  const byCategory = {};
  txs.forEach(t => { byCategory[t.category] = (byCategory[t.category]||0) + t.price; });
  const labels = Object.keys(byCategory);
  const values = Object.values(byCategory);
  const colors = labels.map(l => CAT_COLORS[l] || '#94a3b8');
  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById('categoryChart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets:[{ data:values, backgroundColor:colors.map(c=>c+'99'), borderColor:colors, borderWidth:1.5, hoverBorderWidth:2.5 }]
    },
    options: {
      responsive:true, maintainAspectRatio:false, cutout:'65%',
      plugins: {
        legend: { position:'right', labels:{ color:'#9ba8bb', font:{family:"'DM Sans'",size:11}, padding:12, usePointStyle:true, pointStyleWidth:8 } },
        tooltip: {
          backgroundColor:'#172130', borderColor:'rgba(255,255,255,0.08)', borderWidth:1,
          titleColor:'#e8edf5', bodyColor:'#9ba8bb',
          callbacks: { label: ctx => ` $${ctx.parsed} (${Math.round(ctx.parsed/values.reduce((a,b)=>a+b,0)*100)}%)` }
        }
      }
    }
  });
}
</script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────
# ROUTES  (unchanged logic)
# ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        if u in users and users[u]['password'] == p:
            session['username'] = u
            session['fullname'] = users[u]['fullname']
            return redirect(url_for('dashboard'))
        error = "Invalid username or password."
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        c = request.form.get('confirm', '')
        if not u or not p:
            error = "Username and password are required."
        elif p != c:
            error = "Passwords do not match."
        elif u in users:
            error = "Username already taken."
        else:
            users[u] = {'password': p, 'fullname': fullname}
            session['username'] = u
            session['fullname'] = fullname
            return redirect(url_for('dashboard'))
    return render_template_string(REGISTER_HTML, error=error)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_HTML, username=session.get('fullname') or session['username'])

@app.route('/run', methods=['POST'])
def run():
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        data = request.json
        income = int(data.get('income', 50000))
        max_expense = int(data.get('max_expense', 500))
        out = run_tracker(income, max_expense)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == '__main__':
    print("Starting Vaultex AI at http://127.0.0.1:9876")
    app.run(host='0.0.0.0', debug=True, port=9876)
