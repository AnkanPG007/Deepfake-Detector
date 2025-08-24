import os
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

TEMPLATE_DIR = 'templates'
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

login_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-sm">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Login</h1>
        {% with messages = get_flashed_messages() %}
        {% if messages %}
        <div class="mb-4">
            {% for message in messages %}
            <p class="text-red-500 text-sm text-center">{{ message }}</p>
            {% endfor %}
        </div>
        {% endif %}
        {% endwith %}
        <form method="POST" action="{{ url_for('login') }}">
            <div class="mb-4">
                <label for="username" class="block text-gray-700 font-bold mb-2">Username</label>
                <input type="text" id="username" name="username" class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" required>
            </div>
            <div class="mb-6">
                <label for="password" class="block text-gray-700 font-bold mb-2">Password</label>
                <input type="password" id="password" name="password" class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" required>
            </div>
            <button type="submit" class="w-full bg-blue-600 text-white font-bold py-2 rounded-lg hover:bg-blue-700 transition duration-300">Log In</button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Don't have an account? <a href="{{ url_for('register') }}" class="text-blue-600 hover:underline">Register here</a>
        </p>
    </div>
</body>
</html>
"""

register_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-sm">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Register</h1>
        {% with messages = get_flashed_messages() %}
        {% if messages %}
        <div class="mb-4">
            {% for message in messages %}
            <p class="text-red-500 text-sm text-center">{{ message }}</p>
            {% endfor %}
        </div>
        {% endif %}
        {% endwith %}
        <form method="POST" action="{{ url_for('register') }}">
            <div class="mb-4">
                <label for="username" class="block text-gray-700 font-bold mb-2">Username</label>
                <input type="text" id="username" name="username" class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" required>
            </div>
            <div class="mb-6">
                <label for="password" class="block text-gray-700 font-bold mb-2">Password</label>
                <input type="password" id="password" name="password" class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" required>
            </div>
            <button type="submit" class="w-full bg-green-600 text-white font-bold py-2 rounded-lg hover:bg-green-700 transition duration-300">Register</button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Already have an account? <a href="{{ url_for('login') }}" class="text-blue-600 hover:underline">Log in here</a>
        </p>
    </div>
</body>
</html>
"""

with open(os.path.join(TEMPLATE_DIR, 'login.html'), 'w') as f:
    f.write(login_html_template)
with open(os.path.join(TEMPLATE_DIR, 'register.html'), 'w') as f:
    f.write(register_html_template)

app = Flask(__name__)
app.secret_key = "supersecretkey"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@app.route("/")
def home():
    
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists!")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.")
        return redirect(url_for("login"))

    return render_template_string(register_html_template)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Login successful!")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password")
            return redirect(url_for("login"))
   
    return render_template_string(login_html_template)

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; }}
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center justify-center min-h-screen">
        <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md text-center">
            <h1 class="text-3xl font-bold text-gray-800 mb-4">Welcome, {session['username']}!</h1>
            <p class="text-gray-600 mb-6">You are now logged in.</p>
            <a href="{url_for('logout')}" class="bg-red-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-red-600 transition duration-300">Logout</a>
        </div>
    </body>
    </html>
    """

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))

if __name__ == "__main__":
    if not os.path.exists("users.db"):
        with app.app_context():
            db.create_all()
    app.run(debug=True)
