# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:29:13 2020

@author: Luke
"""

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/<name>")
def hello_name(name):
    return "Hello " + name

if __name__== "__main__":
    app.run(debug=True)