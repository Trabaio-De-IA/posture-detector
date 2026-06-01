from flask import Blueprint, render_template

blueprint = Blueprint("main", __name__)


@blueprint.route("/")
def index():
    return render_template("index.html")


@blueprint.route("/about")
def about():
    return render_template("about.html")


@blueprint.route("/team")
def team():
    members = [
        {"name": "Guilherme", "role": "Desenvolvedor"},
        {"name": "João",      "role": "Desenvolvedor"},
        {"name": "Kevin",     "role": "Desenvolvedor"},
        {"name": "Yane",      "role": "Desenvolvedor"},
    ]
    return render_template("team.html", members=members)
