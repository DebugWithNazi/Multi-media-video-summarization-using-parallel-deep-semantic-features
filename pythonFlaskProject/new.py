from flask import Flask, redirect, url_for, render_template
# instance of a flask web application
app = Flask(__name__)

# 1st page
# @app.route("/")
# def home():
#     return "hello, this is the main page!"
@app.route("/")
def home():
    return render_template("index.html")

# 2nd page
# @app.route("/<name>")
# def user(name):
#     return f"hello {name}!"
@app.route("/<name>")
def user(name):
    return render_template("user.html", content=name)
# 3rd page
@app.route("/admin/")
def admin():
    return redirect(url_for("user", name="Admin"))
# Run the app
if __name__ == "__main__":
    app.run(debug=True)