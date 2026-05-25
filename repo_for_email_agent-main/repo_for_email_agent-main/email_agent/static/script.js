document.addEventListener('DOMContentLoaded', () => {
    const processBtn    = document.getElementById('processBtn');
    const emailInput    = document.getElementById('emailInput');
    const btnText       = document.getElementById('btnText');
    const loader        = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');

    // Result elements
    const priorityBadge = document.getElementById('priorityBadge');
    const categoryBadge = document.getElementById('categoryBadge');
    const reasonText    = document.getElementById('reasonText');
    const replyText     = document.getElementById('replyText');
    const followupText  = document.getElementById('followupText');
    const historyText   = document.getElementById('historyText');

    let currentFolder = 'inbox';

    // Load existing history on page load
    loadHistory();

    // Tab switching logic
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFolder = btn.dataset.folder;
            loadHistory();
        });
    });

    processBtn.addEventListener('click', async () => {
        const email_text = emailInput.value.trim();

        if (!email_text) {
            alert('Please enter an email to process.');
            return;
        }

        // UI Loading state
        processBtn.disabled = true;
        btnText.textContent = 'Processing...';
        loader.style.display = 'block';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/api/process_email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email_text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) throw new Error(data.error);

            // Populate current results
            priorityBadge.textContent = data.priority.toUpperCase();
            priorityBadge.className = `badge badge-priority ${data.priority.toLowerCase()}`;
            categoryBadge.textContent = data.category.toUpperCase();
            reasonText.textContent  = data.reason;
            replyText.textContent   = data.reply;
            followupText.textContent = data.followup;

            // Update history and stay in inbox
            currentFolder = 'inbox';
            document.querySelector('[data-folder="inbox"]').click();

            // Show results
            resultsSection.style.display = 'flex';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the email: ' + error.message);
        } finally {
            processBtn.disabled = false;
            btnText.textContent = 'Process Email';
            loader.style.display = 'none';
        }
    });

    window.viewHistoryItem = (id) => {
        window.location.href = `/email/${id}`;
    };

    window.toggleStar = async (e, id) => {
        e.stopPropagation();
        try {
            const res = await fetch(`/api/toggle_star/${id}`, { method: 'POST' });
            if (res.ok) loadHistory();
        } catch (err) { console.error(err); }
    };

    window.toggleTrash = async (e, id) => {
        e.stopPropagation();
        try {
            const res = await fetch(`/api/toggle_trash/${id}`, { method: 'POST' });
            if (res.ok) loadHistory();
        } catch (err) { console.error(err); }
    };

    window.deletePermanent = async (e, id) => {
        e.stopPropagation();
        if (!confirm('Permanently delete this record?')) return;
        try {
            const res = await fetch(`/api/delete_permanent/${id}`, { method: 'DELETE' });
            if (res.ok) loadHistory();
        } catch (err) { console.error(err); }
    };

    async function loadHistory() {
        try {
            const res = await fetch(`/api/history?folder=${currentFolder}`);
            if (!res.ok) return;
            const data = await res.json();
            renderHistory(data.history);
        } catch (e) {
            console.log('No history loaded:', e);
        }
    }

    function renderHistory(historyArray) {
        if (!historyArray || historyArray.length === 0) {
            historyText.innerHTML = `<div style="text-align:center; padding: 2rem; color: var(--text-muted);">
                <i class="fas fa-folder-open" style="font-size: 2rem; display: block; margin-bottom: 1rem;"></i>
                No emails in ${currentFolder}
            </div>`;
            return;
        }

        let html = '';
        historyArray.forEach((entry) => {
            const priorityClass = entry.priority === 'high' ? '#fca5a5' :
                                  entry.priority === 'medium' ? '#fde047' : '#86efac';
            
            const starClass = entry.is_starred ? 'active' : '';
            const trashIcon = currentFolder === 'trash' ? 'fa-trash-restore restore-btn' : 'fa-trash-alt trash-btn';
            const trashAction = 'toggleTrash';

            html += `
<div class="history-item" onclick="viewHistoryItem(${entry.id})" style="border-left: 3px solid ${priorityClass}; padding: 1rem; margin-bottom: 1rem; background: rgba(255,255,255,0.05); border-radius: 0 12px 12px 0; cursor: pointer; transition: all 0.3s ease; position: relative;">
    <div style="display:flex; justify-content:space-between; align-items: flex-start; margin-bottom:0.5rem;">
        <div>
            <span style="font-weight:700; color:${priorityClass}; font-size: 0.8rem; letter-spacing: 1px;">${(entry.priority || '?').toUpperCase()}</span>
            <span style="color: var(--text-muted); margin: 0 0.5rem;">|</span>
            <span style="font-weight:600; color: var(--text-main); font-size: 0.9rem;">${(entry.category || '?').toUpperCase()}</span>
        </div>
        <div class="history-actions">
            <button class="action-btn star-btn ${starClass}" onclick="toggleStar(event, ${entry.id})">
                <i class="fas fa-star"></i>
            </button>
            <button class="action-btn" onclick="${trashAction}(event, ${entry.id})">
                <i class="fas ${trashIcon}"></i>
            </button>
            ${currentFolder === 'trash' ? `
            <button class="action-btn trash-btn" onclick="deletePermanent(event, ${entry.id})">
                <i class="fas fa-times-circle"></i>
            </button>` : ''}
        </div>
    </div>
    <div style="font-size:0.85rem; color: var(--text-main); margin-bottom:0.4rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity: 0.9;">
        ${entry.email_text ? entry.email_text.substring(0, 80) + '...' : ''}
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size:0.8rem; color: var(--text-muted); font-style: italic;">${entry.reason ? entry.reason.substring(0, 60) + '...' : ''}</div>
        <div style="font-size:0.7rem; color: var(--text-muted);">${entry.processed_at || ''}</div>
    </div>
</div>`;
        });

        historyText.innerHTML = html;
    }
});
