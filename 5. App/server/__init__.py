import os
import sys
from flask import Flask
from flask_cloudy import Storage

app = Flask(__name__)

app.config.update({
    "STORAGE_PROVIDER": "LOCAL",
    "STORAGE_CONTAINER": "./video_data/",
    "STORAGE_KEY": "",
    "STORAGE_SECRET": "",
    "STORAGE_SERVER": True,
    "STORAGE_ALLOWED_EXTENSIONS":["flv", "avi", "rmvb", "mp4"],
    # "STORAGE_SERVER_URL": "/video_data/"
})

storage = Storage()
storage.init_app(app)


@app.context_processor
def inject_title():
    return {'title': "基于YOLO网络的行人检测研究与应用"}
    # 需要返回字典，等同于return dict(title="基于YOLO网络的行人检测研究与应用")

import views, errors

#, commands



# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager
#
# # SQLite URI compatible
# WIN = sys.platform.startswith('win')
# if WIN:
#     prefix = 'sqlite:///'
# else:
#     prefix = 'sqlite:////'

# app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
# app.config['SQLALCHEMY_DATABASE_URI'] = prefix + os.path.join(os.path.dirname(app.root_path), os.getenv('DATABASE_FILE', 'data.db'))
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
# login_manager = LoginManager(app)


# @login_manager.user_loader
# def load_user(user_id):
#     from watchlist.models import User
#     user = User.query.get(int(user_id))
#     return user
#
#
# login_manager.login_view = 'login'
# # login_manager.login_message = 'Your custom message'




