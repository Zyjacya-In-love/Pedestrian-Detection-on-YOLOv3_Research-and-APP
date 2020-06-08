from __init__ import app
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')

if __name__ == '__main__':
    scfg = cfg['server']
    # app.run(host='0.0.0.0', port='5000', debug=True)
    # app.run(host='0.0.0.0', port='5000', threaded=True, ssl_context='adhoc')
    app.run(host=scfg['host'], port=scfg['port'], debug=scfg.getboolean('debug'), threaded=scfg.getboolean('threaded'), ssl_context=None if scfg['ssl_context']=='None' else scfg['ssl_context'])