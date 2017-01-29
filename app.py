# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:35:40 2017

@author: sominwadhwa
"""


from flask import Flask, url_for, request, render_template, redirect, send_file, make_response

app = Flask(__name__)
@app.route('/')
def index():
    return redirect(url_for('Task'))
@app.route('/Task')
def Task():
    return render_template('hello.html')
    