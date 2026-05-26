// Auth Elements
const authOverlay = document.getElementById('authOverlay');
const mainContent = document.getElementById('mainContent');
const authTitle = document.getElementById('authTitle');
const authSubtitle = document.getElementById('authSubtitle');
const authBtn = document.getElementById('authBtn');
const toggleAuth = document.getElementById('toggleAuth');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');
const currentUserLabel = document.getElementById('currentUser');
const logoutBtn = document.getElementById('logoutBtn');

// Main Elements
const topicInput = document.getElementById('topicInput');
const runBtn = document.getElementById('runBtn');
const logsDiv = document.getElementById('logs');
const resultDiv = document.getElementById('result');
const downloadBtn = document.getElementById('downloadBtn');
const toast = document.getElementById('toast');

let logInterval;
let isLoginMode = true;

// Auth Logic
async function handleAuth() {
    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();

    if (!username || !password) return showToast('Fill all fields', '#ef4444');

    const endpoint = isLoginMode ? '/login' : '/register';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();

        if (data.status === 'success') {
            if (isLoginMode) {
                loginUser(username);
            } else {
                showToast('Registered! Please login.');
                toggleAuthMode();
            }
        } else {
            showToast(data.message || 'Auth failed', '#ef4444');
        }
    } catch (error) {
        showToast('Server error', '#ef4444');
    }
}

function loginUser(username) {
    localStorage.setItem('researchUser', username);
    currentUserLabel.innerText = username;
    authOverlay.style.display = 'none';
    mainContent.style.display = 'block';
    showToast(`Welcome, ${username}!`);
}

function logoutUser() {
    localStorage.removeItem('researchUser');
    authOverlay.style.display = 'flex';
    mainContent.style.display = 'none';
    usernameInput.value = '';
    passwordInput.value = '';
}

function toggleAuthMode() {
    isLoginMode = !isLoginMode;
    authTitle.innerText = isLoginMode ? 'Welcome Back' : 'Join the Lab';
    authSubtitle.innerText = isLoginMode ? 'Please login to access the research lab' : 'Create an account to start exploring';
    authBtn.innerText = isLoginMode ? 'Login' : 'Register';
    toggleAuth.innerText = isLoginMode ? 'Register' : 'Login';
}

// Check for existing session
window.onload = () => {
    const user = localStorage.getItem('researchUser');
    if (user) {
        loginUser(user);
    }
};

// Research Logic
async function startResearch() {
    const topic = topicInput.value.trim();
    if (!topic) return showToast('Please enter a topic', '#ef4444');

    topicInput.disabled = true;
    runBtn.disabled = true;
    runBtn.innerText = 'Researching...';
    logsDiv.innerHTML = '';
    resultDiv.innerHTML = '<div class="placeholder-text">Processing your request...</div>';

    try {
        const response = await fetch('/run-research', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === 'started') {
                showToast('Research started successfully!');
                startLogPolling();
            } else {
                throw new Error(data.message || 'Unexpected server response');
            }
        } else {
            const err = await response.json().catch(() => ({}));
            throw new Error(`Server error ${response.status}: ${err.detail || response.statusText}`);
        }
    } catch (error) {
        showToast(error.message, '#ef4444');
        addLogEntry(`❌ Error: ${error.message}`);
        resetUI();
    }
}

function startLogPolling() {
    logInterval = setInterval(async () => {
        try {
            const response = await fetch('/logs');
            const data = await response.json();
            
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => {
                    if (log.startsWith('FINAL_RESULT:')) {
                        clearInterval(logInterval);
                        displayResult(log.replace('FINAL_RESULT:', ''));
                        resetUI();
                    } else if (log.startsWith('SYSTEM_ERROR:')) {
                        clearInterval(logInterval);
                        const errMsg = log.replace('SYSTEM_ERROR:', '').trim().split('\n')[0];
                        addLogEntry(`❌ Research failed: ${errMsg}`);
                        showToast('Research failed. See Live Progress for details.', '#ef4444');
                        resetUI();
                    } else {
                        addLogEntry(log);
                    }
                });
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

function addLogEntry(text) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerText = text;
    logsDiv.appendChild(entry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

function displayResult(text) {
    resultDiv.innerHTML = `
        <div class="report-header">
            <h1 style="color: #6366f1; margin-bottom: 20px;">AI Research Report</h1>
            <p style="color: #94a3b8; margin-bottom: 30px;">Generated on: ${new Date().toLocaleString()}</p>
        </div>
        <div class="final-result">${text}</div>
    `;
    downloadBtn.style.display = 'block';
}

function downloadPDF() {
    const element = document.getElementById('result');
    showToast('Preparing your PDF...');
    element.classList.add('pdf-export');
    
    const opt = {
        margin: [0.5, 0.5],
        filename: `Research_Report_${Date.now()}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, letterRendering: true, backgroundColor: '#ffffff' },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    html2pdf().set(opt).from(element).save()
        .then(() => {
            element.classList.remove('pdf-export');
            showToast('PDF downloaded!');
        })
        .catch(err => {
            element.classList.remove('pdf-export');
            showToast('Failed to generate PDF', '#ef4444');
        });
}

function resetUI() {
    topicInput.disabled = false;
    runBtn.disabled = false;
    runBtn.innerText = 'Start Research';
}

function showToast(msg, color = '#10b981') {
    toast.innerText = msg;
    toast.style.background = color;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// Event Listeners
authBtn.addEventListener('click', handleAuth);
toggleAuth.addEventListener('click', toggleAuthMode);
logoutBtn.addEventListener('click', logoutUser);
runBtn.addEventListener('click', startResearch);
downloadBtn.addEventListener('click', downloadPDF);
topicInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') startResearch();
});
passwordInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleAuth();
});
