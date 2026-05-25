from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import agent_email_1

app = Flask(__name__)
app.config['SECRET_KEY'] = 'email-agent-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///email_agent.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# ─────────────────────────────────────────────
#  DATABASE MODELS
# ─────────────────────────────────────────────

class User(UserMixin, db.Model):
    """Stores registered users."""
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship: one user → many email history records
    email_history = db.relationship('EmailHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class EmailHistory(db.Model):
    """Stores every email processed by a user — persisted in SQLite."""
    __tablename__ = 'email_history'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    email_text  = db.Column(db.Text, nullable=False)
    priority    = db.Column(db.String(20))
    category    = db.Column(db.String(50))
    reason      = db.Column(db.Text)
    reply       = db.Column(db.Text)
    followup    = db.Column(db.Text)
    is_starred  = db.Column(db.Boolean, default=False)
    is_deleted  = db.Column(db.Boolean, default=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':           self.id,
            'email_text':   self.email_text,
            'priority':     self.priority,
            'category':     self.category,
            'reason':       self.reason,
            'reply':        self.reply,
            'followup':     self.followup,
            'is_starred':   self.is_starred,
            'is_deleted':   self.is_deleted,
            'processed_at': self.processed_at.strftime('%Y-%m-%d %H:%M:%S')
        }


# Create tables on first run
with app.app_context():
    db.create_all()


# ─────────────────────────────────────────────
#  AUTH LOADER
# ─────────────────────────────────────────────

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route('/')
@login_required
def home():
    return render_template('index.html', username=current_user.username)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ─────────────────────────────────────────────
#  API ROUTES
# ─────────────────────────────────────────────

@app.route('/api/process_email', methods=['POST'])
@login_required
def process_email():
    data = request.json
    email_text = data.get('email_text', '').strip()

    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    try:
        priority, category, reason, reply, followup = agent_email_1.process_email_full(email_text)

        # ── Persist to SQLite ──────────────────────────
        record = EmailHistory(
            user_id    = current_user.id,
            email_text = email_text,
            priority   = priority,
            category   = category,
            reason     = reason,
            reply      = reply,
            followup   = followup
        )
        db.session.add(record)
        db.session.commit()
        # ───────────────────────────────────────────────

        # Return all history records for this user (newest first)
        all_history = (
            EmailHistory.query
            .filter_by(user_id=current_user.id)
            .order_by(EmailHistory.processed_at.desc())
            .all()
        )

        return jsonify({
            'priority': priority,
            'category': category,
            'reason':   reason,
            'reply':    reply,
            'followup': followup,
            'history':  [h.to_dict() for h in all_history]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Fetch conversation history for the logged-in user with optional folder filtering."""
    folder = request.args.get('folder', 'inbox')
    
    query = EmailHistory.query.filter_by(user_id=current_user.id)
    
    if folder == 'starred':
        query = query.filter_by(is_starred=True, is_deleted=False)
    elif folder == 'trash':
        query = query.filter_by(is_deleted=True)
    else: # inbox
        query = query.filter_by(is_deleted=False)
        
    records = query.order_by(EmailHistory.processed_at.desc()).all()
    return jsonify({'history': [r.to_dict() for r in records], 'count': len(records)})

@app.route('/api/toggle_star/<int:record_id>', methods=['POST'])
@login_required
def toggle_star(record_id):
    record = EmailHistory.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    record.is_starred = not record.is_starred
    db.session.commit()
    return jsonify({'is_starred': record.is_starred})

@app.route('/api/toggle_trash/<int:record_id>', methods=['POST'])
@login_required
def toggle_trash(record_id):
    record = EmailHistory.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    record.is_deleted = not record.is_deleted
    db.session.commit()
    return jsonify({'is_deleted': record.is_deleted})

@app.route('/api/delete_permanent/<int:record_id>', methods=['DELETE'])
@login_required
def delete_permanent(record_id):
    record = EmailHistory.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/email/<int:record_id>')
@login_required
def view_email_details(record_id):
    """View details of a specific email on a new page."""
    record = EmailHistory.query.get_or_404(record_id)
    
    # Ensure the user owns this record
    if record.user_id != current_user.id:
        return redirect(url_for('home'))
        
    return render_template('details.html', email=record, username=current_user.username)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
